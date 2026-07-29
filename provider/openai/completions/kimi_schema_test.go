package completions

import (
	"reflect"
	"strings"
	"testing"
)

func TestNormalizeSchemaForKimiDistributesMemohAttachmentObject(t *testing.T) {
	original := kimiMemohAttachmentSchema()
	originalCopy := cloneJSONValue(original)

	normalizedValue, err := normalizeSchemaForKimi(original)
	if err != nil {
		t.Fatalf("normalizeSchemaForKimi: %v", err)
	}
	normalized, ok := normalizedValue.(map[string]any)
	if !ok {
		t.Fatalf("normalized schema = %T, want map[string]any", normalizedValue)
	}
	if !reflect.DeepEqual(original, originalCopy) {
		t.Fatal("normalization mutated the input schema")
	}

	objectBranch := kimiAttachmentObjectBranch(t, normalized)
	for _, key := range []string{"type", "properties", "required", "additionalProperties"} {
		if _, exists := objectBranch[key]; exists {
			t.Fatalf("attachment object parent still contains %q: %#v", key, objectBranch)
		}
	}
	innerAnyOf := schemaAnyOf(t, objectBranch, "attachment object")
	if len(innerAnyOf) != 5 {
		t.Fatalf("inner anyOf length = %d, want 5", len(innerAnyOf))
	}

	for index, rawBranch := range innerAnyOf {
		branch, ok := rawBranch.(map[string]any)
		if !ok {
			t.Fatalf("inner anyOf[%d] = %T, want map[string]any", index, rawBranch)
		}
		if branch["type"] != "object" {
			t.Fatalf("inner anyOf[%d].type = %#v, want object", index, branch["type"])
		}
		if branch["additionalProperties"] != false {
			t.Fatalf("inner anyOf[%d].additionalProperties = %#v, want false", index, branch["additionalProperties"])
		}
		properties, ok := branch["properties"].(map[string]any)
		if !ok {
			t.Fatalf("inner anyOf[%d].properties = %T, want map[string]any", index, branch["properties"])
		}
		for _, name := range []string{"path", "url", "base64", "content_hash", "platform_key", "metadata"} {
			if _, exists := properties[name]; !exists {
				t.Fatalf("inner anyOf[%d] is missing property %q", index, name)
			}
		}
		required, err := schemaStringArray(branch["required"], "required")
		if err != nil {
			t.Fatalf("inner anyOf[%d].required: %v", index, err)
		}
		if len(required) != 1 {
			t.Fatalf("inner anyOf[%d].required = %#v, want one item", index, required)
		}
		if _, exists := properties[required[0]]; !exists {
			t.Fatalf("inner anyOf[%d] requires undefined property %q", index, required[0])
		}
	}

	normalizedAgain, err := normalizeSchemaForKimi(normalized)
	if err != nil {
		t.Fatalf("second normalizeSchemaForKimi: %v", err)
	}
	if !reflect.DeepEqual(normalized, normalizedAgain) {
		t.Fatalf("normalization is not idempotent:\nfirst:  %#v\nsecond: %#v", normalized, normalizedAgain)
	}
}

func TestNormalizeSchemaForKimiRejectsUnsafeObjectBranchMerge(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string"},
		},
		"anyOf": []any{
			map[string]any{
				"properties": map[string]any{
					"url": map[string]any{"type": "string"},
				},
				"required": []string{"url"},
			},
		},
	}

	_, err := normalizeSchemaForKimi(schema)
	if err == nil {
		t.Fatal("expected unsafe properties merge to fail")
	}
	if !strings.Contains(err.Error(), "$.anyOf[0].properties") {
		t.Fatalf("error %q does not identify the conflicting schema path", err)
	}
}

func TestNormalizeSchemaForKimiRejectsRequiredPropertyMissingFromProperties(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"url": map[string]any{"type": "string"},
		},
		"anyOf": []any{
			map[string]any{"required": []string{"path"}},
		},
	}

	_, err := normalizeSchemaForKimi(schema)
	if err == nil {
		t.Fatal("expected undefined required property to fail")
	}
	if !strings.Contains(err.Error(), `required property "path" is not defined in properties`) {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestNormalizeSchemaForKimiRecursesIntoCombinators(t *testing.T) {
	nestedAnyOf := func() map[string]any {
		return map[string]any{
			"type": "string",
			"anyOf": []any{
				map[string]any{"minLength": float64(1)},
				map[string]any{"type": "string", "maxLength": float64(8)},
			},
		}
	}
	assertNormalized := func(t *testing.T, schema map[string]any, label string) {
		t.Helper()
		if _, exists := schema["type"]; exists {
			t.Fatalf("%s: parent type was not pushed into anyOf branches: %#v", label, schema)
		}
		anyOf := schemaAnyOf(t, schema, label)
		for index, rawBranch := range anyOf {
			branch, ok := rawBranch.(map[string]any)
			if !ok {
				t.Fatalf("%s.anyOf[%d] = %T, want map[string]any", label, index, rawBranch)
			}
			if branch["type"] != "string" {
				t.Fatalf("%s.anyOf[%d].type = %#v, want string", label, index, branch["type"])
			}
		}
	}

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"value": map[string]any{
				"oneOf": []any{nestedAnyOf()},
				"allOf": []any{nestedAnyOf()},
				"not":   nestedAnyOf(),
			},
		},
	}

	normalizedValue, err := normalizeSchemaForKimi(schema)
	if err != nil {
		t.Fatalf("normalizeSchemaForKimi: %v", err)
	}
	normalized, ok := normalizedValue.(map[string]any)
	if !ok {
		t.Fatalf("normalized schema = %T, want map[string]any", normalizedValue)
	}

	value := schemaPathMap(t, normalized, "properties", "value")
	for _, key := range []string{"oneOf", "allOf"} {
		list, ok := value[key].([]any)
		if !ok || len(list) != 1 {
			t.Fatalf("value.%s = %#v, want one branch", key, value[key])
		}
		branch, ok := list[0].(map[string]any)
		if !ok {
			t.Fatalf("value.%s[0] = %T, want map[string]any", key, list[0])
		}
		assertNormalized(t, branch, "value."+key+"[0]")
	}
	not, ok := value["not"].(map[string]any)
	if !ok {
		t.Fatalf("value.not = %T, want map[string]any", value["not"])
	}
	assertNormalized(t, not, "value.not")
}

func TestNormalizeSchemaForKimiRejectsBooleanCombinatorBranches(t *testing.T) {
	schema := map[string]any{
		"oneOf": []any{true},
	}

	_, err := normalizeSchemaForKimi(schema)
	if err == nil {
		t.Fatal("expected boolean oneOf branch to fail")
	}
	if !strings.Contains(err.Error(), "$.oneOf[0]") {
		t.Fatalf("error %q does not identify the offending schema path", err)
	}
}

func TestNormalizeSchemaForKimiRejectsNonObjectRootSchema(t *testing.T) {
	cases := []struct {
		name    string
		value   any
		wantErr string
	}{
		{"boolean", true, "$: boolean schemas are not supported"},
		{"array", []any{"a"}, "$: expected a schema object"},
		{"string", "not a schema", "$: expected a schema object"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := normalizeSchemaForKimi(tc.value)
			if err == nil {
				t.Fatalf("expected %T root schema to fail", tc.value)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error %q does not contain %q", err, tc.wantErr)
			}
		})
	}
}

func TestNormalizeSchemaForKimiRejectsInvalidAdditionalProperties(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"config": map[string]any{
				"type":                 "object",
				"additionalProperties": "yes",
			},
		},
	}

	_, err := normalizeSchemaForKimi(schema)
	if err == nil {
		t.Fatal("expected invalid additionalProperties to fail")
	}
	if !strings.Contains(err.Error(), "$.properties.config.additionalProperties") {
		t.Fatalf("error %q does not identify the offending schema path", err)
	}
}

func schemaPathMap(t *testing.T, m map[string]any, path ...string) map[string]any {
	t.Helper()
	current := m
	for _, key := range path {
		next, ok := current[key].(map[string]any)
		if !ok {
			t.Fatalf("schema path %v: %q = %T, want map[string]any", path, key, current[key])
		}
		current = next
	}
	return current
}

func kimiMemohAttachmentSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"attachments": map[string]any{
				"type": "array",
				"items": map[string]any{
					"anyOf": []any{
						map[string]any{"type": "string"},
						map[string]any{
							"type":                 "object",
							"additionalProperties": false,
							"anyOf": []any{
								map[string]any{"required": []string{"path"}},
								map[string]any{"required": []string{"url"}},
								map[string]any{"required": []string{"base64"}},
								map[string]any{"required": []string{"content_hash"}},
								map[string]any{"required": []string{"platform_key"}},
							},
							"properties": map[string]any{
								"path":         map[string]any{"type": "string"},
								"url":          map[string]any{"type": "string"},
								"base64":       map[string]any{"type": "string"},
								"content_hash": map[string]any{"type": "string"},
								"platform_key": map[string]any{"type": "string"},
								"metadata":     map[string]any{"type": "object"},
							},
						},
					},
				},
			},
		},
		"required": []string{"attachments"},
	}
}

func kimiAttachmentObjectBranch(t *testing.T, schema map[string]any) map[string]any {
	t.Helper()
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("root properties = %T, want map[string]any", schema["properties"])
	}
	attachments, ok := properties["attachments"].(map[string]any)
	if !ok {
		t.Fatalf("attachments = %T, want map[string]any", properties["attachments"])
	}
	items, ok := attachments["items"].(map[string]any)
	if !ok {
		t.Fatalf("attachments.items = %T, want map[string]any", attachments["items"])
	}
	outerAnyOf := schemaAnyOf(t, items, "attachments.items")
	if len(outerAnyOf) != 2 {
		t.Fatalf("outer anyOf length = %d, want 2", len(outerAnyOf))
	}
	objectBranch, ok := outerAnyOf[1].(map[string]any)
	if !ok {
		t.Fatalf("outer anyOf[1] = %T, want map[string]any", outerAnyOf[1])
	}
	return objectBranch
}

func schemaAnyOf(t *testing.T, schema map[string]any, label string) []any {
	t.Helper()
	anyOf, ok := schema["anyOf"].([]any)
	if !ok {
		t.Fatalf("%s.anyOf = %T, want []any", label, schema["anyOf"])
	}
	return anyOf
}
