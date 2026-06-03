package openai

import "strings"

// NormalizeReasoningEffort maps effort strings that are not accepted by the
// OpenAI wire format into the closest supported equivalent. Currently this
// rewrites "max" → "xhigh" because OpenAI-format endpoints reject "max".
//
// NOTE: The Memoh server also filters "max" out of the selectable effort list
// before it reaches the SDK, so this is a defence-in-depth measure.
func NormalizeReasoningEffort(effort string) string {
	if strings.EqualFold(strings.TrimSpace(effort), "max") {
		return "xhigh"
	}
	return effort
}
