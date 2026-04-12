package copilot

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/memohai/twilight-ai/sdk"
)

type streamProcessor struct {
	ctx                context.Context
	ch                 chan sdk.StreamPart
	textStartSent      bool
	reasoningStartSent bool
	rawFinishReason    string
	finishReason       sdk.FinishReason
	usage              sdk.Usage
	chunkID            string
	chunkModel         string
	chunkCreated       int64
	flushed            bool
	pendingToolCalls   map[int]*streamingToolCall
}

func (sp *streamProcessor) send(part sdk.StreamPart) bool {
	select {
	case sp.ch <- part:
		return true
	case <-sp.ctx.Done():
		return false
	}
}

func (sp *streamProcessor) endReasoning(id string) {
	if sp.reasoningStartSent {
		sp.send(&sdk.ReasoningEndPart{ID: id})
		sp.reasoningStartSent = false
	}
}

func (sp *streamProcessor) endText(id string) {
	if sp.textStartSent {
		sp.send(&sdk.TextEndPart{ID: id})
		sp.textStartSent = false
	}
}

func (sp *streamProcessor) flush() {
	if sp.flushed {
		return
	}
	sp.flushed = true
	sp.endReasoning(sp.chunkID)
	sp.endText(sp.chunkID)
	for _, stc := range sp.pendingToolCalls {
		sp.finishToolCall(stc)
	}
}

func (sp *streamProcessor) finishToolCall(stc *streamingToolCall) {
	if stc.finished {
		return
	}
	sp.send(&sdk.ToolInputEndPart{ID: stc.id})
	var input any
	if err := json.Unmarshal([]byte(stc.args), &input); err != nil {
		sp.send(&sdk.ErrorPart{Error: fmt.Errorf("github-copilot: unmarshal tool call arguments for %q: %w", stc.name, err)})
	}
	sp.send(&sdk.StreamToolCallPart{
		ToolCallID: stc.id,
		ToolName:   stc.name,
		Input:      input,
	})
	stc.finished = true
}

func (sp *streamProcessor) processChunk(chunk *chatChunkResponse) error {
	if sp.chunkID == "" {
		sp.chunkID = chunk.ID
		sp.chunkModel = chunk.Model
		sp.chunkCreated = chunk.Created
	}

	if chunk.Usage != nil {
		sp.usage = convertUsage(chunk.Usage)
	}

	if len(chunk.Choices) == 0 {
		return nil
	}
	choice := chunk.Choices[0]

	sp.processReasoning(&choice.Delta, chunk.ID)
	sp.processContent(&choice.Delta, chunk.ID)
	sp.processToolCallDeltas(choice.Delta.ToolCalls, chunk.ID)
	sp.processImages(choice.Delta.Images, chunk.ID)
	sp.processFinishReason(&choice)

	return nil
}

func (sp *streamProcessor) processReasoning(delta *chatChunkDelta, chunkID string) {
	reasoningContent := reasoningFromDelta(delta)
	if reasoningContent == "" {
		return
	}
	if !sp.reasoningStartSent {
		sp.send(&sdk.ReasoningStartPart{ID: chunkID})
		sp.reasoningStartSent = true
	}
	sp.send(&sdk.ReasoningDeltaPart{ID: chunkID, Text: reasoningContent})
}

func (sp *streamProcessor) processContent(delta *chatChunkDelta, chunkID string) {
	if delta.Content == "" {
		return
	}
	sp.endReasoning(chunkID)
	if !sp.textStartSent {
		sp.send(&sdk.TextStartPart{ID: chunkID})
		sp.textStartSent = true
	}
	sp.send(&sdk.TextDeltaPart{ID: chunkID, Text: delta.Content})
}

func (sp *streamProcessor) processToolCallDeltas(toolCalls []chatToolCallChunk, chunkID string) {
	if len(toolCalls) == 0 {
		return
	}
	sp.endReasoning(chunkID)
	sp.endText(chunkID)

	for _, tc := range toolCalls {
		idx := tc.Index
		stc, exists := sp.pendingToolCalls[idx]
		if !exists {
			id := tc.ID
			if id == "" {
				id = generateID()
			}
			stc = &streamingToolCall{id: id, name: tc.Function.Name}
			sp.pendingToolCalls[idx] = stc
			sp.send(&sdk.ToolInputStartPart{
				ID:       stc.id,
				ToolName: stc.name,
			})
		}
		if tc.Function.Arguments != "" {
			stc.args += tc.Function.Arguments
			sp.send(&sdk.ToolInputDeltaPart{
				ID:    stc.id,
				Delta: tc.Function.Arguments,
			})

			if !stc.finished && json.Valid([]byte(stc.args)) {
				sp.finishToolCall(stc)
			}
		}
	}
}

func (sp *streamProcessor) processImages(images []chatImagePart, chunkID string) {
	for _, img := range images {
		url := img.ImageURL.URL
		if url == "" {
			continue
		}
		sp.endText(chunkID)
		sp.endReasoning(chunkID)
		mediaType, data := parseDataURL(url)
		sp.send(&sdk.StreamFilePart{
			File: sdk.GeneratedFile{
				Data:      data,
				MediaType: mediaType,
			},
		})
	}
}

func (sp *streamProcessor) processFinishReason(choice *chatChunkChoice) {
	if choice.FinishReason == nil || *choice.FinishReason == "" {
		return
	}
	sp.rawFinishReason = *choice.FinishReason
	sp.finishReason = mapFinishReason(sp.rawFinishReason)

	sp.flush()

	sp.send(&sdk.FinishStepPart{
		FinishReason:    sp.finishReason,
		RawFinishReason: sp.rawFinishReason,
		Usage:           sp.usage,
		Response: sdk.ResponseMetadata{
			ID:        sp.chunkID,
			ModelID:   sp.chunkModel,
			Timestamp: time.Unix(sp.chunkCreated, 0),
		},
	})
}
