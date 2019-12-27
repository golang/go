package source

import (
	"go/ast"

	"golang.org/x/tools/internal/lsp/protocol"

	errors "golang.org/x/xerrors"
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

// keyword looks at the current scope of an *ast.Ident and recommends keywords
func (c *completer) keyword() error {
	keywordScore := float64(0.9)
	if _, ok := c.path[0].(*ast.Ident); !ok {
		// TODO(golang/go#34009): Support keyword completion in any context
		return errors.Errorf("keywords are currently only recommended for identifiers")
	}
	// Track which keywords we've already determined are in a valid scope
	// Use score to order keywords by how close we are to where they are useful
	valid := make(map[string]float64)

	// only suggest keywords at the begnning of a statement
	switch c.path[1].(type) {
	case *ast.BlockStmt, *ast.CommClause, *ast.CaseClause, *ast.ExprStmt:
	default:
		return nil
	}

	// Filter out keywords depending on scope
	// Skip the first one because we want to look at the enclosing scopes
	path := c.path[1:]
	for i, n := range path {
		switch node := n.(type) {
		case *ast.CaseClause:
			// only recommend "fallthrough" and "break" within the bodies of a case clause
			if c.pos > node.Colon {
				valid[BREAK] = keywordScore
				// "fallthrough" is only valid in switch statements.
				// A case clause is always nested within a block statement in a switch statement,
				// that block statement is nested within either a TypeSwitchStmt or a SwitchStmt.
				if i+2 >= len(path) {
					continue
				}
				if _, ok := path[i+2].(*ast.SwitchStmt); ok {
					valid[FALLTHROUGH] = keywordScore
				}
			}
		case *ast.CommClause:
			if c.pos > node.Colon {
				valid[BREAK] = keywordScore
			}
		case *ast.TypeSwitchStmt, *ast.SelectStmt, *ast.SwitchStmt:
			valid[CASE] = keywordScore + lowScore
			valid[DEFAULT] = keywordScore + lowScore
		case *ast.ForStmt:
			valid[BREAK] = keywordScore
			valid[CONTINUE] = keywordScore
		// This is a bit weak, functions allow for many keywords
		case *ast.FuncDecl:
			if node.Body != nil && c.pos > node.Body.Lbrace {
				valid[DEFER] = keywordScore - lowScore
				valid[RETURN] = keywordScore - lowScore
				valid[FOR] = keywordScore - lowScore
				valid[GO] = keywordScore - lowScore
				valid[SWITCH] = keywordScore - lowScore
				valid[SELECT] = keywordScore - lowScore
				valid[IF] = keywordScore - lowScore
				valid[ELSE] = keywordScore - lowScore
				valid[VAR] = keywordScore - lowScore
				valid[CONST] = keywordScore - lowScore
			}
		}
	}

	for ident, score := range valid {
		if c.matcher.Score(ident) > 0 {
			c.items = append(c.items, CompletionItem{
				Label:      ident,
				Kind:       protocol.KeywordCompletion,
				InsertText: ident,
				Score:      score,
			})
		}
	}
	return nil
}
