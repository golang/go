package syntax

// RewriteExtensionsPreTypes2 runs extension rewrites that are safe to do before the first
// types2 pass in the B1 enum pipeline.
//
// Critical: this intentionally leaves enum syntax intact (enum declarations, constructors,
// and match patterns), so types2 can perform inference and we can later lower enums using
// that information.
//
// Practically, this means we skip the main rewriter pass (`RewriteQuestionExprs`) because
// it performs enum lowering (rewriteEnums + enum match rewriting + ctor sugar).
func RewriteExtensionsPreTypes2(file *File) map[string]*OverloadInfo {
	if file == nil {
		return nil
	}
	// Do NOT call RewriteQuestionExprs(file) here.
	RewriteMagicAndArithmetic(file)
	overloads := RewriteInitAndOverloads(file)
	RewriteMethodDecorators(file)
	RewriteDefaultParams(file)
	return overloads
}


