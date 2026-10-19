# Section 5: lexer, parser, AST (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 224-313 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

**SECTION 5: LEXER, PARSER, AND AST NODE SPECIFICATION (`src/cmd/compile/internal/syntax`):**

**5.1. Token Table Additions (`syntax/tokens.go`):**
Zero new globally reserved keywords are added to the keyword hash table. The scanner (`syntax/scanner.go`) adds exactly 5 punctuation/literal tokens and 2 contextual statement identifiers:

| Token Constant | Lexical Form | Scanner State Transition (`scanner.next()`) |
| :--- | :--- | :--- |
| `_Question` | `?` | Triggered on `?` when next char is neither `.` nor `?`. Suppresses automatic semicolon insertion (`nlsemi = true` if postfix). |
| `_QuestionDot` | `?.` | Triggered on `?` followed immediately by `.`. Sets `nlsemi = false`. |
| `_QuestionQuestion` | `??` | Triggered on `?` followed immediately by `?`. Binary operator precedence `1` (below `||`). |
| `_Arrow` | `=>` | Triggered on `=` followed immediately by `>`. Used in single-expression lambdas and `switch` value cases. |
| `_InterpStringLit` | `$"..."` | Triggered on `$` followed immediately by `"`. Lexes balanced `{ expr }` sub-token streams inside string spans. |

**5.2. Concrete AST Struct Definitions (`syntax/nodes_logos.go`):**

```go
// 1. Postfix Error Propagation: expr?
type TryExpr struct {
    X expr
    expr
}

// 2. Optional Chaining & Coalescing: a?.b, a ?? b
type OptSelectorExpr struct {
    X   expr
    Sel *Name
    expr
}

// 3. Error-Returning & Error-Only Defer: defer fn()?, errdefer fn()
type DeferStmt struct {
    Call      expr
    Propagate bool // true for 'defer fn()?'
    ErrOnly   bool // true for 'errdefer fn()'
    ScopeOnly bool // true for 'scope defer fn()'
    stmt
}

// 4. Compile-Time Control Flow: const if, const for, const switch
type ConstIfStmt struct {
    Cond expr
    Then *BlockStmt
    Else stmt // *ConstIfStmt or *BlockStmt or nil
    stmt
}

type ConstForStmt struct {
    Index *Name // optional '_' or ident
    Field *Name // loop variable 'f'
    X     expr  // comptime.Fields[T]()
    Body  *BlockStmt
    stmt
}

type ConstSwitchStmt struct {
    Targ   expr // Generic type parameter identifier 'T'
    Cases  []*ConstCaseClause
    stmt
}

// 5. Compile-Time Field Projection: target.[f]
type FieldProjExpr struct {
    X     expr
    Field *Name
    expr
}

// 6. Interface Structural Property Declaration: const Name string; HP int
type FieldContractDecl struct {
    IsConst bool
    Name    *Name
    Type    expr
    decl
}
```

**5.3. LL(1) Non-Backtracking Disambiguation Algorithms (`syntax/parser.go`):**
* **Rule A (`ParenExpr` vs `TupleType` vs `ArrowLambda`):**
  When the parser encounters `(`, it calls `p.parseParenOrTupleOrLambda()` with zero token backtracking:
  1. Parse the first element using `p.exprOrType()`.
  2. If the next token is `:`, parse as a **Named Tuple Type** `(name1 Type1, name2 Type2)`.
  3. If the next token is `,`, parse the comma-separated list into `elems []Expr`. After `) `, inspect `p.tok`:
     * If `p.tok == _Arrow` (`=>`), validate that all `elems` are `*Name` identifiers and return `*ArrowFuncLit{Params: elems, Body: p.expr()}`.
     * Otherwise, return `*TupleLit{Elems: elems}` (or `*TupleType` in type context).
  4. If `len(elems) == 1` and `p.tok == _Arrow`, return `*ArrowFuncLit`. Otherwise return standard `*ParenExpr`.
* **Rule B (`errdefer` Contextual Parsing in `p.stmtOrDecl()`):**
  When `p.tok == _Name && p.lit == "errdefer"`, the parser peeks 1 token ahead (`p.peek()`). If the next token starts an expression (`_Name`, `_Lparen`, etc.) rather than `:=`, `=`, `,`, or `:`, it consumes `errdefer` and returns `&DeferStmt{Call: p.expr(), ErrOnly: true}`. Thus existing Go code `errdefer := 1` parses as a standard assignment.

---

