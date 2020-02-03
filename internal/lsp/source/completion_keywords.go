package source

import (
	"go/ast"

	"golang.org/x/tools/internal/lsp/protocol"
)

const (
	BREAK       = "break"
	CASE        = "case"
	CHAN        = "chan"
	CONST       = "const"
	CONTINUE    = "continue"
	DEFAULT     = "default"
	DEFER       = "defer"
	ELSE        = "else"
	FALLTHROUGH = "fallthrough"
	FOR         = "for"
	FUNC        = "func"
	GO          = "go"
	GOTO        = "goto"
	IF          = "if"
	IMPORT      = "import"
	INTERFACE   = "interface"
	MAP         = "map"
	PACKAGE     = "package"
	RANGE       = "range"
	RETURN      = "return"
	SELECT      = "select"
	STRUCT      = "struct"
	SWITCH      = "switch"
	TYPE        = "type"
	VAR         = "var"
)

// addKeywordCompletions offers keyword candidates appropriate at the position.
func (c *completer) addKeywordCompletions() {
	const keywordScore = 0.9

	seen := make(map[string]bool)

	// addKeywords dedupes and adds completion items for the specified
	// keywords with the specified score.
	addKeywords := func(score float64, kws ...string) {
		for _, kw := range kws {
			if seen[kw] {
				continue
			}
			seen[kw] = true

			if c.matcher.Score(kw) > 0 {
				c.items = append(c.items, CompletionItem{
					Label:      kw,
					Kind:       protocol.KeywordCompletion,
					InsertText: kw,
					Score:      score,
				})
			}
		}
	}

	// If we are at the file scope, only offer decl keywords. We don't
	// get *ast.Idents at the file scope because non-keyword identifiers
	// turn into *ast.BadDecl, not *ast.Ident.
	if len(c.path) == 1 || isASTFile(c.path[1]) {
		addKeywords(keywordScore, TYPE, CONST, VAR, FUNC, IMPORT)
		return
	} else if _, ok := c.path[0].(*ast.Ident); !ok {
		// Otherwise only offer keywords if the client is completing an identifier.
		return
	}

	// Only suggest keywords if we are beginning a statement.
	switch c.path[1].(type) {
	case *ast.BlockStmt, *ast.CommClause, *ast.CaseClause, *ast.ExprStmt:
	default:
		return
	}

	// Filter out keywords depending on scope
	// Skip the first one because we want to look at the enclosing scopes
	path := c.path[1:]
	for i, n := range path {
		switch node := n.(type) {
		case *ast.CaseClause:
			// only recommend "fallthrough" and "break" within the bodies of a case clause
			if c.pos > node.Colon {
				addKeywords(keywordScore, BREAK)
				// "fallthrough" is only valid in switch statements.
				// A case clause is always nested within a block statement in a switch statement,
				// that block statement is nested within either a TypeSwitchStmt or a SwitchStmt.
				if i+2 >= len(path) {
					continue
				}
				if _, ok := path[i+2].(*ast.SwitchStmt); ok {
					addKeywords(keywordScore, FALLTHROUGH)
				}
			}
		case *ast.CommClause:
			if c.pos > node.Colon {
				addKeywords(keywordScore, BREAK)
			}
		case *ast.TypeSwitchStmt, *ast.SelectStmt, *ast.SwitchStmt:
			addKeywords(keywordScore+lowScore, CASE, DEFAULT)
		case *ast.ForStmt:
			addKeywords(keywordScore, BREAK, CONTINUE)
		// This is a bit weak, functions allow for many keywords
		case *ast.FuncDecl:
			if node.Body != nil && c.pos > node.Body.Lbrace {
				addKeywords(keywordScore-lowScore, DEFER, RETURN, FOR, GO, SWITCH, SELECT, IF, ELSE, VAR, CONST, GOTO, TYPE)
			}
		}
	}

}
