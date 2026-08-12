package sdk

import "reflect"

func cloneDeferredToolApprovals(src []DeferredToolApproval) []DeferredToolApproval {
	if src == nil {
		return nil
	}
	return cloneSDKValue(src).([]DeferredToolApproval)
}

func clonePauseMessages(src []Message) []Message {
	if src == nil {
		return nil
	}
	return cloneSDKValue(src).([]Message)
}

func cloneToolCalls(src []ToolCall) []ToolCall {
	if src == nil {
		return nil
	}
	return cloneSDKValue(src).([]ToolCall)
}

// deferredWithOriginalCalls keeps approval results returned by the host while
// restoring each pending call from the provider snapshot. Approval callbacks
// retain their historical ability to mutate nested input values, but those
// mutations must not rewrite the portable record of what the model requested.
func deferredWithOriginalCalls(deferred []DeferredToolApproval, original []ToolCall) []DeferredToolApproval {
	byID := make(map[string]ToolCall, len(original))
	for _, call := range original {
		byID[call.ToolCallID] = call
	}
	result := make([]DeferredToolApproval, len(deferred))
	for i, approval := range deferred {
		result[i] = approval
		if call, ok := byID[approval.ToolCall.ToolCallID]; ok {
			result[i].ToolCall = call
		}
	}
	return result
}

func cloneAnyMap(src map[string]any) map[string]any {
	if src == nil {
		return nil
	}
	return cloneSDKValue(src).(map[string]any)
}

func cloneStepOutcome(src stepOutcome) stepOutcome {
	cloned := src
	cloned.reasoningMeta = cloneAnyMap(src.reasoningMeta)
	cloned.toolCalls = cloneToolCalls(src.toolCalls)
	return cloned
}

// Pause data crosses a persistence boundary. Clone reference-bearing values
// recursively for the SDK's supported, persistable data: JSON-shaped values
// and typed containers/structs whose mutable state is held in exported fields.
// Opaque state such as funcs, channels, unsafe pointers, and references hidden
// in unexported fields remains shared.
//
// As with any handoff in Go, the producer must stop mutating a value while or
// after returning it to the SDK. A copy can isolate later consumers; it cannot
// make a concurrent read of a map safe while its producer is still writing.
func cloneSDKValue(value any) any {
	if value == nil {
		return nil
	}
	return clonePauseReflect(reflect.ValueOf(value), make(map[pauseCloneVisit]reflect.Value)).Interface()
}

type pauseCloneVisit struct {
	typ  reflect.Type
	kind reflect.Kind
	ptr  uintptr
	len  int
	cap  int
}

func clonePauseReflect(value reflect.Value, visited map[pauseCloneVisit]reflect.Value) reflect.Value {
	if !value.IsValid() {
		return value
	}

	switch value.Kind() {
	case reflect.Interface:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		cloned := reflect.New(value.Type()).Elem()
		cloned.Set(clonePauseReflect(value.Elem(), visited))
		return cloned

	case reflect.Pointer:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		visit := pauseCloneVisit{typ: value.Type(), kind: value.Kind(), ptr: value.Pointer()}
		if cloned, ok := visited[visit]; ok {
			return cloned
		}
		cloned := reflect.New(value.Type().Elem())
		visited[visit] = cloned
		cloned.Elem().Set(clonePauseReflect(value.Elem(), visited))
		return cloned

	case reflect.Map:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		visit := pauseCloneVisit{typ: value.Type(), kind: value.Kind(), ptr: value.Pointer()}
		if cloned, ok := visited[visit]; ok {
			return cloned
		}
		cloned := reflect.MakeMapWithSize(value.Type(), value.Len())
		visited[visit] = cloned
		iter := value.MapRange()
		for iter.Next() {
			key := clonePauseReflect(iter.Key(), visited)
			item := clonePauseReflect(iter.Value(), visited)
			cloned.SetMapIndex(key, item)
		}
		return cloned

	case reflect.Slice:
		if value.IsNil() {
			return reflect.Zero(value.Type())
		}
		visit := pauseCloneVisit{
			typ: value.Type(), kind: value.Kind(), ptr: value.Pointer(),
			len: value.Len(), cap: value.Cap(),
		}
		if cloned, ok := visited[visit]; ok {
			return cloned
		}
		cloned := reflect.MakeSlice(value.Type(), value.Len(), value.Len())
		visited[visit] = cloned
		for i := 0; i < value.Len(); i++ {
			cloned.Index(i).Set(clonePauseReflect(value.Index(i), visited))
		}
		return cloned

	case reflect.Array:
		cloned := reflect.New(value.Type()).Elem()
		for i := 0; i < value.Len(); i++ {
			cloned.Index(i).Set(clonePauseReflect(value.Index(i), visited))
		}
		return cloned

	case reflect.Struct:
		// Start with a value copy so opaque, unexported state remains valid.
		// Exported fields carry the portable data and are recursively isolated.
		cloned := reflect.New(value.Type()).Elem()
		cloned.Set(value)
		for i := 0; i < value.NumField(); i++ {
			if value.Type().Field(i).PkgPath != "" {
				continue
			}
			cloned.Field(i).Set(clonePauseReflect(value.Field(i), visited))
		}
		return cloned

	default:
		return value
	}
}
