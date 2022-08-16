// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gopls_test

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

// verify that type errors in Initialize lsp messages don't cause
// any other unmarshalling errors. The code looks at single values and the
// first component of array values. Each occurrence is replaced by something
// of a different type,  the resulting string unmarshalled, and compared to
// the unmarshalling of the unchanged strings. The test passes if there is no
// more than a single difference reported. That is, if changing a single value
// in the message changes no more than a single value in the unmarshalled struct,
// it is safe to ignore *json.UnmarshalTypeError.

// strings are changed to numbers or bools (true)
// bools are changed to numbers or strings
// numbers are changed to strings or bools

// a recent Initialize message taken from a log
const input = `{"processId":38349,"clientInfo":{"name":"vscode","version":"1.56.0-insider"},"rootPath":"/Users/pjw/latest/tools","rootUri":"file:///Users/pjw/latest/tools","capabilities":{"workspace":{"applyEdit":true,"workspaceEdit":{"documentChanges":true,"resourceOperations":["create","rename","delete"],"failureHandling":"textOnlyTransactional"},"didChangeConfiguration":{"dynamicRegistration":true},"didChangeWatchedFiles":{"dynamicRegistration":true},"symbol":{"dynamicRegistration":true,"symbolKind":{"valueSet":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]},"tagSupport":{"valueSet":[1]}},"executeCommand":{"dynamicRegistration":true},"configuration":true,"workspaceFolders":true,"semanticTokens":{"refreshSupport":true}},"textDocument":{"publishDiagnostics":{"relatedInformation":true,"versionSupport":false,"tagSupport":{"valueSet":[1,2]},"codeDescriptionSupport":true,"dataSupport":true},"synchronization":{"dynamicRegistration":true,"willSave":true,"willSaveWaitUntil":true,"didSave":true},"completion":{"dynamicRegistration":true,"contextSupport":true,"completionItem":{"snippetSupport":true,"commitCharactersSupport":true,"documentationFormat":["markdown","plaintext"],"deprecatedSupport":true,"preselectSupport":true,"tagSupport":{"valueSet":[1]},"insertReplaceSupport":true,"resolveSupport":{"properties":["documentation","detail","additionalTextEdits"]}},"completionItemKind":{"valueSet":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]}},"hover":{"dynamicRegistration":true,"contentFormat":["markdown","plaintext"]},"signatureHelp":{"dynamicRegistration":true,"signatureInformation":{"documentationFormat":["markdown","plaintext"],"parameterInformation":{"labelOffsetSupport":true},"activeParameterSupport":true},"contextSupport":true},"definition":{"dynamicRegistration":true,"linkSupport":true},"references":{"dynamicRegistration":true},"documentHighlight":{"dynamicRegistration":true},"documentSymbol":{"dynamicRegistration":true,"symbolKind":{"valueSet":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]},"hierarchicalDocumentSymbolSupport":true,"tagSupport":{"valueSet":[1]},"labelSupport":true},"codeAction":{"dynamicRegistration":true,"isPreferredSupport":true,"disabledSupport":true,"dataSupport":true,"resolveSupport":{"properties":["edit"]},"codeActionLiteralSupport":{"codeActionKind":{"valueSet":["","quickfix","refactor","refactor.extract","refactor.inline","refactor.rewrite","source","source.organizeImports"]}}},"codeLens":{"dynamicRegistration":true},"formatting":{"dynamicRegistration":true},"rangeFormatting":{"dynamicRegistration":true},"onTypeFormatting":{"dynamicRegistration":true},"rename":{"dynamicRegistration":true,"prepareSupport":true,"prepareSupportDefaultBehavior":true},"documentLink":{"dynamicRegistration":true,"tooltipSupport":true},"typeDefinition":{"dynamicRegistration":true,"linkSupport":true},"implementation":{"dynamicRegistration":true,"linkSupport":true},"colorProvider":{"dynamicRegistration":true},"foldingRange":{"dynamicRegistration":true,"rangeLimit":5000,"lineFoldingOnly":true},"declaration":{"dynamicRegistration":true,"linkSupport":true},"selectionRange":{"dynamicRegistration":true},"callHierarchy":{"dynamicRegistration":true},"semanticTokens":{"dynamicRegistration":true,"tokenTypes":["namespace","type","class","enum","interface","struct","typeParameter","parameter","variable","property","enumMember","event","function","member","macro","keyword","modifier","comment","string","number","regexp","operator"],"tokenModifiers":["declaration","definition","readonly","static","deprecated","abstract","async","modification","documentation","defaultLibrary"],"formats":["relative"],"requests":{"range":true,"full":{"delta":true}}}},"window":{"workDoneProgress":true}},"initializationOptions":{"usePlaceholders":true,"completionDocumentation":true,"verboseOutput":false,"codelenses":{"gc_details":true},"analyses":{"fillstruct":true,"staticcheck":true},"experimentalWorkspaceModule":true,"semanticTokens":true},"trace":"off","workspaceFolders":[{"uri":"file:///Users/pjw/latest/tools","name":"tools"}]}`

type DiffReporter struct {
	path  cmp.Path
	diffs []string
}

func (r *DiffReporter) PushStep(ps cmp.PathStep) {
	r.path = append(r.path, ps)
}

func (r *DiffReporter) Report(rs cmp.Result) {
	if !rs.Equal() {
		vx, vy := r.path.Last().Values()
		r.diffs = append(r.diffs, fmt.Sprintf("%#v:\n\t-: %+v\n\t+: %+v\n", r.path, vx, vy))
	}
}

func (r *DiffReporter) PopStep() {
	r.path = r.path[:len(r.path)-1]
}

func (r *DiffReporter) String() string {
	return strings.Join(r.diffs, "\n")
}

func TestStringChanges(t *testing.T) {
	// string as value
	stringLeaf := regexp.MustCompile(`:("[^"]*")`)
	leafs := stringLeaf.FindAllStringSubmatchIndex(input, -1)
	allDeltas(t, leafs, "23", "true")
	// string as first element of array
	stringArray := regexp.MustCompile(`[[]("[^"]*")`)
	arrays := stringArray.FindAllStringSubmatchIndex(input, -1)
	allDeltas(t, arrays, "23", "true")
}

func TestBoolChanges(t *testing.T) {
	boolLeaf := regexp.MustCompile(`:(true|false)(,|})`)
	leafs := boolLeaf.FindAllStringSubmatchIndex(input, -1)
	allDeltas(t, leafs, "23", `"xx"`)
	boolArray := regexp.MustCompile(`:[[](true|false)(,|])`)
	arrays := boolArray.FindAllStringSubmatchIndex(input, -1)
	allDeltas(t, arrays, "23", `"xx"`)
}

func TestNumberChanges(t *testing.T) {
	numLeaf := regexp.MustCompile(`:(\d+)(,|})`)
	leafs := numLeaf.FindAllStringSubmatchIndex(input, -1)
	allDeltas(t, leafs, "true", `"xx"`)
	numArray := regexp.MustCompile(`:[[](\d+)(,|])`)
	arrays := numArray.FindAllStringSubmatchIndex(input, -1)
	allDeltas(t, arrays, "true", `"xx"`)
}

// v is a set of matches. check that substituting any repl never
// creates more than 1 unmarshaling error
func allDeltas(t *testing.T, v [][]int, repls ...string) {
	t.Helper()
	for _, repl := range repls {
		for i, x := range v {
			err := tryChange(x[2], x[3], repl)
			if err != nil {
				t.Errorf("%d:%q %v", i, input[x[2]:x[3]], err)
			}
		}
	}
}

func tryChange(start, end int, repl string) error {
	var p, q protocol.ParamInitialize
	mod := input[:start] + repl + input[end:]
	excerpt := func() (string, string) {
		a := start - 5
		if a < 0 {
			a = 0
		}
		b := end + 5
		if b > len(input) {
			// trusting repl to be no longer than what it replaces
			b = len(input)
		}
		ma := input[a:b]
		mb := mod[a:b]
		return ma, mb
	}
	if err := json.Unmarshal([]byte(input), &p); err != nil {
		return fmt.Errorf("%s %v", repl, err)
	}
	if err := json.Unmarshal([]byte(mod), &q); err == nil {
		return nil // no errors is ok
	} else if _, ok := err.(*json.UnmarshalTypeError); !ok {
		return fmt.Errorf("%T, not *json.UnmarshalTypeError", err)
	}

	var r DiffReporter
	cmp.Diff(p, q, cmp.Reporter(&r))
	if len(r.diffs) > 1 { // 0 is possible, e.g., for interface{}
		ma, mb := excerpt()
		return fmt.Errorf("got %d diffs for %q\n%s\n%s", len(r.diffs), repl, ma, mb)
	}
	return nil
}
