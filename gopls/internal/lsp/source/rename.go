// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/diff"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/refactor/satisfy"
)

type renamer struct {
	ctx                context.Context
	snapshot           Snapshot
	refs               []*ReferenceInfo
	objsToUpdate       map[types.Object]bool
	hadConflicts       bool
	errors             string
	from, to           string
	satisfyConstraints map[satisfy.Constraint]bool
	packages           map[*types.Package]Package // may include additional packages that are a dep of pkg.
	msets              typeutil.MethodSetCache
	changeMethods      bool
}

type PrepareItem struct {
	Range protocol.Range
	Text  string
}

// PrepareRename searches for a valid renaming at position pp.
//
// The returned usererr is intended to be displayed to the user to explain why
// the prepare fails. Probably we could eliminate the redundancy in returning
// two errors, but for now this is done defensively.
func PrepareRename(ctx context.Context, snapshot Snapshot, f FileHandle, pp protocol.Position) (_ *PrepareItem, usererr, err error) {
	// Find position of the package name declaration.
	ctx, done := event.Start(ctx, "source.PrepareRename")
	defer done()

	// Is the cursor within the package name declaration?
	if pgf, inPackageName, err := parsePackageNameDecl(ctx, snapshot, f, pp); err != nil {
		return nil, err, err
	} else if inPackageName {
		item, err := prepareRenamePackageName(ctx, snapshot, pgf)
		return item, err, err
	}

	// Ordinary (non-package) renaming.
	//
	// Type-check the current package, locate the reference at the position,
	// validate the object, and report its name and range.
	//
	// TODO(adonovan): in all cases below, we return usererr=nil,
	// which means we return (nil, nil) at the protocol
	// layer. This seems like a bug, or at best an exploitation of
	// knowledge of VSCode-specific behavior. Can we avoid that?
	pkg, pgf, err := PackageForFile(ctx, snapshot, f.URI(), TypecheckFull, NarrowestPackage)
	if err != nil {
		return nil, nil, err
	}
	pos, err := pgf.PositionPos(pp)
	if err != nil {
		return nil, nil, err
	}
	targets, node, err := objectsAt(pkg.GetTypesInfo(), pgf.File, pos)
	if err != nil {
		return nil, nil, err
	}
	var obj types.Object
	for obj = range targets {
		break // pick one arbitrarily
	}
	if err := checkRenamable(obj); err != nil {
		return nil, nil, err
	}
	rng, err := pgf.NodeRange(node)
	if err != nil {
		return nil, nil, err
	}
	if _, isImport := node.(*ast.ImportSpec); isImport {
		// We're not really renaming the import path.
		rng.End = rng.Start
	}
	return &PrepareItem{
		Range: rng,
		Text:  obj.Name(),
	}, nil, nil
}

func prepareRenamePackageName(ctx context.Context, snapshot Snapshot, pgf *ParsedGoFile) (*PrepareItem, error) {
	// Does the client support file renaming?
	fileRenameSupported := false
	for _, op := range snapshot.View().Options().SupportedResourceOperations {
		if op == protocol.Rename {
			fileRenameSupported = true
			break
		}
	}
	if !fileRenameSupported {
		return nil, errors.New("can't rename package: LSP client does not support file renaming")
	}

	// Check validity of the metadata for the file's containing package.
	fileMeta, err := snapshot.MetadataForFile(ctx, pgf.URI)
	if err != nil {
		return nil, err
	}
	if len(fileMeta) == 0 {
		return nil, fmt.Errorf("no packages found for file %q", pgf.URI)
	}
	meta := fileMeta[0]
	if meta.Name == "main" {
		return nil, fmt.Errorf("can't rename package \"main\"")
	}
	if strings.HasSuffix(string(meta.Name), "_test") {
		return nil, fmt.Errorf("can't rename x_test packages")
	}
	if meta.Module == nil {
		return nil, fmt.Errorf("can't rename package: missing module information for package %q", meta.PkgPath)
	}
	if meta.Module.Path == string(meta.PkgPath) {
		return nil, fmt.Errorf("can't rename package: package path %q is the same as module path %q", meta.PkgPath, meta.Module.Path)
	}

	// Return the location of the package declaration.
	rng, err := pgf.NodeRange(pgf.File.Name)
	if err != nil {
		return nil, err
	}
	return &PrepareItem{
		Range: rng,
		Text:  string(meta.Name),
	}, nil
}

func checkRenamable(obj types.Object) error {
	if v, ok := obj.(*types.Var); ok && v.Embedded() {
		return errors.New("can't rename embedded fields: rename the type directly or name the field")
	}
	if obj.Name() == "_" {
		return errors.New("can't rename \"_\"")
	}
	return nil
}

// Rename returns a map of TextEdits for each file modified when renaming a
// given identifier within a package and a boolean value of true for renaming
// package and false otherwise.
func Rename(ctx context.Context, s Snapshot, f FileHandle, pp protocol.Position, newName string) (map[span.URI][]protocol.TextEdit, bool, error) {
	ctx, done := event.Start(ctx, "source.Rename")
	defer done()

	// Cursor within package name declaration?
	if _, inPackageName, err := parsePackageNameDecl(ctx, s, f, pp); err != nil {
		return nil, false, err
	} else if inPackageName {
		return renamePackageName(ctx, s, f, pp, newName)
	}

	// ordinary (non-package) rename
	qos, err := qualifiedObjsAtProtocolPos(ctx, s, f.URI(), pp)
	if err != nil {
		return nil, false, err
	}
	if err := checkRenamable(qos[0].obj); err != nil {
		return nil, false, err
	}
	if qos[0].obj.Name() == newName {
		return nil, false, fmt.Errorf("old and new names are the same: %s", newName)
	}
	if !isValidIdentifier(newName) {
		return nil, false, fmt.Errorf("invalid identifier to rename: %q", newName)
	}
	result, err := renameObj(ctx, s, newName, qos)
	if err != nil {
		return nil, false, err
	}

	return result, false, nil
}

func renamePackageName(ctx context.Context, s Snapshot, f FileHandle, pp protocol.Position, newName string) (map[span.URI][]protocol.TextEdit, bool, error) {
	if !isValidIdentifier(newName) {
		return nil, true, fmt.Errorf("%q is not a valid identifier", newName)
	}

	fileMeta, err := s.MetadataForFile(ctx, f.URI())
	if err != nil {
		return nil, true, err
	}

	if len(fileMeta) == 0 {
		return nil, true, fmt.Errorf("no packages found for file %q", f.URI())
	}

	// We need metadata for the relevant package and module paths. These should
	// be the same for all packages containing the file.
	//
	// TODO(rfindley): we mix package path and import path here haphazardly.
	// Fix this.
	meta := fileMeta[0]
	oldPath := meta.PkgPath
	var modulePath PackagePath
	if mi := meta.Module; mi == nil {
		return nil, true, fmt.Errorf("cannot rename package: missing module information for package %q", meta.PkgPath)
	} else {
		modulePath = PackagePath(mi.Path)
	}

	if strings.HasSuffix(newName, "_test") {
		return nil, true, fmt.Errorf("cannot rename to _test package")
	}

	metadata, err := s.AllMetadata(ctx)
	if err != nil {
		return nil, true, err
	}

	renamingEdits, err := renamePackage(ctx, s, modulePath, oldPath, PackageName(newName), metadata)
	if err != nil {
		return nil, true, err
	}

	oldBase := filepath.Dir(span.URI.Filename(f.URI()))
	newPkgDir := filepath.Join(filepath.Dir(oldBase), newName)

	// TODO: should this operate on all go.mod files, irrespective of whether they are included in the workspace?
	// Get all active mod files in the workspace
	modFiles := s.ModFiles()
	for _, m := range modFiles {
		fh, err := s.GetFile(ctx, m)
		if err != nil {
			return nil, true, err
		}
		pm, err := s.ParseMod(ctx, fh)
		if err != nil {
			return nil, true, err
		}

		modFileDir := filepath.Dir(pm.URI.Filename())
		affectedReplaces := []*modfile.Replace{}

		// Check if any replace directives need to be fixed
		for _, r := range pm.File.Replace {
			if !strings.HasPrefix(r.New.Path, "/") && !strings.HasPrefix(r.New.Path, "./") && !strings.HasPrefix(r.New.Path, "../") {
				continue
			}

			replacedPath := r.New.Path
			if strings.HasPrefix(r.New.Path, "./") || strings.HasPrefix(r.New.Path, "../") {
				replacedPath = filepath.Join(modFileDir, r.New.Path)
			}

			// TODO: Is there a risk of converting a '\' delimited replacement to a '/' delimited replacement?
			if !strings.HasPrefix(filepath.ToSlash(replacedPath)+"/", filepath.ToSlash(oldBase)+"/") {
				continue // not affected by the package renaming
			}

			affectedReplaces = append(affectedReplaces, r)
		}

		if len(affectedReplaces) == 0 {
			continue
		}
		copied, err := modfile.Parse("", pm.Mapper.Content, nil)
		if err != nil {
			return nil, true, err
		}

		for _, r := range affectedReplaces {
			replacedPath := r.New.Path
			if strings.HasPrefix(r.New.Path, "./") || strings.HasPrefix(r.New.Path, "../") {
				replacedPath = filepath.Join(modFileDir, r.New.Path)
			}

			suffix := strings.TrimPrefix(replacedPath, string(oldBase))

			newReplacedPath, err := filepath.Rel(modFileDir, newPkgDir+suffix)
			if err != nil {
				return nil, true, err
			}

			newReplacedPath = filepath.ToSlash(newReplacedPath)

			if !strings.HasPrefix(newReplacedPath, "/") && !strings.HasPrefix(newReplacedPath, "../") {
				newReplacedPath = "./" + newReplacedPath
			}

			if err := copied.AddReplace(r.Old.Path, "", newReplacedPath, ""); err != nil {
				return nil, true, err
			}
		}

		copied.Cleanup()
		newContent, err := copied.Format()
		if err != nil {
			return nil, true, err
		}

		// Calculate the edits to be made due to the change.
		diff := s.View().Options().ComputeEdits(string(pm.Mapper.Content), string(newContent))
		modFileEdits, err := ToProtocolEdits(pm.Mapper, diff)
		if err != nil {
			return nil, true, err
		}

		renamingEdits[pm.URI] = append(renamingEdits[pm.URI], modFileEdits...)
	}

	return renamingEdits, true, nil
}

// renamePackage computes all workspace edits required to rename the package
// described by the given metadata, to newName, by renaming its package
// directory.
//
// It updates package clauses and import paths for the renamed package as well
// as any other packages affected by the directory renaming among packages
// described by allMetadata.
func renamePackage(ctx context.Context, s Snapshot, modulePath, oldPath PackagePath, newName PackageName, allMetadata []*Metadata) (map[span.URI][]protocol.TextEdit, error) {
	if modulePath == oldPath {
		return nil, fmt.Errorf("cannot rename package: module path %q is the same as the package path, so renaming the package directory would have no effect", modulePath)
	}

	newPathPrefix := path.Join(path.Dir(string(oldPath)), string(newName))

	edits := make(map[span.URI][]protocol.TextEdit)
	seen := make(seenPackageRename) // track per-file import renaming we've already processed

	// Rename imports to the renamed package from other packages.
	for _, m := range allMetadata {
		// Special case: x_test packages for the renamed package will not have the
		// package path as as a dir prefix, but still need their package clauses
		// renamed.
		if m.PkgPath == oldPath+"_test" {
			newTestName := newName + "_test"

			if err := renamePackageClause(ctx, m, s, newTestName, seen, edits); err != nil {
				return nil, err
			}
			continue
		}

		// Subtle: check this condition before checking for valid module info
		// below, because we should not fail this operation if unrelated packages
		// lack module info.
		if !strings.HasPrefix(string(m.PkgPath)+"/", string(oldPath)+"/") {
			continue // not affected by the package renaming
		}

		if m.Module == nil {
			// This check will always fail under Bazel.
			return nil, fmt.Errorf("cannot rename package: missing module information for package %q", m.PkgPath)
		}

		if modulePath != PackagePath(m.Module.Path) {
			continue // don't edit imports if nested package and renaming package have different module paths
		}

		// Renaming a package consists of changing its import path and package name.
		suffix := strings.TrimPrefix(string(m.PkgPath), string(oldPath))
		newPath := newPathPrefix + suffix

		pkgName := m.Name
		if m.PkgPath == PackagePath(oldPath) {
			pkgName = newName

			if err := renamePackageClause(ctx, m, s, newName, seen, edits); err != nil {
				return nil, err
			}
		}

		imp := ImportPath(newPath) // TODO(adonovan): what if newPath has vendor/ prefix?
		if err := renameImports(ctx, s, m, imp, pkgName, seen, edits); err != nil {
			return nil, err
		}
	}

	return edits, nil
}

// seenPackageRename tracks import path renamings that have already been
// processed.
//
// Due to test variants, files may appear multiple times in the reverse
// transitive closure of a renamed package, or in the reverse transitive
// closure of different variants of a renamed package (both are possible).
// However, in all cases the resulting edits will be the same.
type seenPackageRename map[seenPackageKey]bool
type seenPackageKey struct {
	uri  span.URI
	path PackagePath
}

// add reports whether uri and importPath have been seen, and records them as
// seen if not.
func (s seenPackageRename) add(uri span.URI, path PackagePath) bool {
	key := seenPackageKey{uri, path}
	seen := s[key]
	if !seen {
		s[key] = true
	}
	return seen
}

// renamePackageClause computes edits renaming the package clause of files in
// the package described by the given metadata, to newName.
//
// As files may belong to multiple packages, the seen map tracks files whose
// package clause has already been updated, to prevent duplicate edits.
//
// Edits are written into the edits map.
func renamePackageClause(ctx context.Context, m *Metadata, snapshot Snapshot, newName PackageName, seen seenPackageRename, edits map[span.URI][]protocol.TextEdit) error {
	// Rename internal references to the package in the renaming package.
	for _, uri := range m.CompiledGoFiles {
		if seen.add(uri, m.PkgPath) {
			continue
		}
		fh, err := snapshot.GetFile(ctx, uri)
		if err != nil {
			return err
		}
		f, err := snapshot.ParseGo(ctx, fh, ParseHeader)
		if err != nil {
			return err
		}
		if f.File.Name == nil {
			continue // no package declaration
		}
		rng, err := f.NodeRange(f.File.Name)
		if err != nil {
			return err
		}
		edits[f.URI] = append(edits[f.URI], protocol.TextEdit{
			Range:   rng,
			NewText: string(newName),
		})
	}

	return nil
}

// renameImports computes the set of edits to imports resulting from renaming
// the package described by the given metadata, to a package with import path
// newPath and name newName.
//
// Edits are written into the edits map.
func renameImports(ctx context.Context, snapshot Snapshot, m *Metadata, newPath ImportPath, newName PackageName, seen seenPackageRename, edits map[span.URI][]protocol.TextEdit) error {
	rdeps, err := snapshot.ReverseDependencies(ctx, m.ID, false) // find direct importers
	if err != nil {
		return err
	}

	// Pass 1: rename import paths in import declarations.
	needsTypeCheck := make(map[PackageID][]span.URI)
	for _, rdep := range rdeps {
		if rdep.IsIntermediateTestVariant() {
			continue // for renaming, these variants are redundant
		}

		for _, uri := range rdep.CompiledGoFiles {
			if seen.add(uri, m.PkgPath) {
				continue
			}
			fh, err := snapshot.GetFile(ctx, uri)
			if err != nil {
				return err
			}
			f, err := snapshot.ParseGo(ctx, fh, ParseHeader)
			if err != nil {
				return err
			}
			if f.File.Name == nil {
				continue // no package declaration
			}
			for _, imp := range f.File.Imports {
				if rdep.DepsByImpPath[UnquoteImportPath(imp)] != m.ID {
					continue // not the import we're looking for
				}

				// If the import does not explicitly specify
				// a local name, then we need to invoke the
				// type checker to locate references to update.
				//
				// TODO(adonovan): is this actually true?
				// Renaming an import with a local name can still
				// cause conflicts: shadowing of built-ins, or of
				// package-level decls in the same or another file.
				if imp.Name == nil {
					needsTypeCheck[rdep.ID] = append(needsTypeCheck[rdep.ID], uri)
				}

				// Create text edit for the import path (string literal).
				rng, err := f.NodeRange(imp.Path)
				if err != nil {
					return err
				}
				edits[uri] = append(edits[uri], protocol.TextEdit{
					Range:   rng,
					NewText: strconv.Quote(string(newPath)),
				})
			}
		}
	}

	// If the imported package's name hasn't changed,
	// we don't need to rename references within each file.
	if newName == m.Name {
		return nil
	}

	// Pass 2: rename local name (types.PkgName) of imported
	// package throughout one or more files of the package.
	ids := make([]PackageID, 0, len(needsTypeCheck))
	for id := range needsTypeCheck {
		ids = append(ids, id)
	}
	pkgs, err := snapshot.TypeCheck(ctx, TypecheckFull, ids...)
	if err != nil {
		return err
	}
	for i, id := range ids {
		pkg := pkgs[i]
		for _, uri := range needsTypeCheck[id] {
			f, err := pkg.File(uri)
			if err != nil {
				return err
			}
			for _, imp := range f.File.Imports {
				if imp.Name != nil {
					continue // has explicit local name
				}
				if rdeps[id].DepsByImpPath[UnquoteImportPath(imp)] != m.ID {
					continue // not the import we're looking for
				}

				pkgname := pkg.GetTypesInfo().Implicits[imp].(*types.PkgName)
				qos := []qualifiedObject{{obj: pkgname, pkg: pkg}}

				pkgScope := pkg.GetTypes().Scope()
				fileScope := pkg.GetTypesInfo().Scopes[f.File]

				localName := string(newName)
				try := 0

				// Keep trying with fresh names until one succeeds.
				for fileScope.Lookup(localName) != nil || pkgScope.Lookup(localName) != nil {
					try++
					localName = fmt.Sprintf("%s%d", newName, try)
				}

				// renameObj detects various conflicts, including:
				// - new name conflicts with a package-level decl in this file;
				// - new name hides a package-level decl in another file that
				//   is actually referenced in this file;
				// - new name hides a built-in that is actually referenced
				//   in this file;
				// - a reference in this file to the old package name would
				//   become shadowed by an intervening declaration that
				//   uses the new name.
				// It returns the edits if no conflict was detected.
				//
				// TODO(adonovan): reduce the strength of this operation
				// since, for imports specifically, it should require only
				// the current file and the current package, which we
				// already have. Finding references is trivial (Info.Uses).
				changes, err := renameObj(ctx, snapshot, localName, qos)
				if err != nil {
					return err
				}

				// If the chosen local package name matches the package's
				// new name, delete the change that would have inserted
				// an explicit local name, which is always the lexically
				// first change.
				if localName == string(newName) {
					v := changes[uri]
					sort.Slice(v, func(i, j int) bool {
						return protocol.CompareRange(v[i].Range, v[j].Range) < 0
					})
					changes[uri] = v[1:]
				}
				for uri, changeEdits := range changes {
					edits[uri] = append(edits[uri], changeEdits...)
				}
			}
		}
	}
	return nil
}

// renameObj returns a map of TextEdits for renaming an identifier within a file
// and boolean value of true if there is no renaming conflicts and false otherwise.
func renameObj(ctx context.Context, s Snapshot, newName string, qos []qualifiedObject) (map[span.URI][]protocol.TextEdit, error) {
	refs, err := references(ctx, s, qos)
	if err != nil {
		return nil, err
	}
	r := renamer{
		ctx:          ctx,
		snapshot:     s,
		refs:         refs,
		objsToUpdate: make(map[types.Object]bool),
		from:         qos[0].obj.Name(),
		to:           newName,
		packages:     make(map[*types.Package]Package),
	}

	// A renaming initiated at an interface method indicates the
	// intention to rename abstract and concrete methods as needed
	// to preserve assignability.
	for _, ref := range refs {
		if obj, ok := ref.obj.(*types.Func); ok {
			recv := obj.Type().(*types.Signature).Recv()
			if recv != nil && types.IsInterface(recv.Type().Underlying()) {
				r.changeMethods = true
				break
			}
		}
	}
	for _, from := range refs {
		r.packages[from.pkg.GetTypes()] = from.pkg
	}

	// Check that the renaming of the identifier is ok.
	for _, ref := range refs {
		r.check(ref.obj)
		if r.hadConflicts { // one error is enough.
			break
		}
	}
	if r.hadConflicts {
		return nil, fmt.Errorf("%s", r.errors)
	}

	changes, err := r.update()
	if err != nil {
		return nil, err
	}

	result := make(map[span.URI][]protocol.TextEdit)
	for uri, edits := range changes {
		// These edits should really be associated with FileHandles for maximal correctness.
		// For now, this is good enough.
		fh, err := s.GetFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		data, err := fh.Read()
		if err != nil {
			return nil, err
		}
		m := protocol.NewMapper(uri, data)
		protocolEdits, err := ToProtocolEdits(m, edits)
		if err != nil {
			return nil, err
		}
		result[uri] = protocolEdits
	}
	return result, nil
}

// Rename all references to the identifier.
func (r *renamer) update() (map[span.URI][]diff.Edit, error) {
	result := make(map[span.URI][]diff.Edit)
	seen := make(map[span.Span]bool)

	docRegexp, err := regexp.Compile(`\b` + r.from + `\b`)
	if err != nil {
		return nil, err
	}
	for _, ref := range r.refs {
		refSpan := ref.MappedRange.Span()
		if seen[refSpan] {
			continue
		}
		seen[refSpan] = true

		// Renaming a types.PkgName may result in the addition or removal of an identifier,
		// so we deal with this separately.
		if pkgName, ok := ref.obj.(*types.PkgName); ok && ref.isDeclaration {
			edit, err := r.updatePkgName(pkgName)
			if err != nil {
				return nil, err
			}
			result[refSpan.URI()] = append(result[refSpan.URI()], *edit)
			continue
		}

		// Replace the identifier with r.to.
		edit := diff.Edit{
			Start: refSpan.Start().Offset(),
			End:   refSpan.End().Offset(),
			New:   r.to,
		}

		result[refSpan.URI()] = append(result[refSpan.URI()], edit)

		if !ref.isDeclaration || ref.ident == nil { // uses do not have doc comments to update.
			continue
		}

		doc := r.docComment(ref.pkg, ref.ident)
		if doc == nil {
			continue
		}

		// Perform the rename in doc comments declared in the original package.
		// go/parser strips out \r\n returns from the comment text, so go
		// line-by-line through the comment text to get the correct positions.
		for _, comment := range doc.List {
			if isDirective(comment.Text) {
				continue
			}
			// TODO(adonovan): why are we looping over lines?
			// Just run the loop body once over the entire multiline comment.
			lines := strings.Split(comment.Text, "\n")
			tokFile := ref.pkg.FileSet().File(comment.Pos())
			commentLine := tokFile.Line(comment.Pos())
			uri := span.URIFromPath(tokFile.Name())
			for i, line := range lines {
				lineStart := comment.Pos()
				if i > 0 {
					lineStart = tokFile.LineStart(commentLine + i)
				}
				for _, locs := range docRegexp.FindAllIndex([]byte(line), -1) {
					// The File.Offset static check complains
					// even though these uses are manifestly safe.
					start, end, _ := safetoken.Offsets(tokFile, lineStart+token.Pos(locs[0]), lineStart+token.Pos(locs[1]))
					result[uri] = append(result[uri], diff.Edit{
						Start: start,
						End:   end,
						New:   r.to,
					})
				}
			}
		}
	}

	return result, nil
}

// docComment returns the doc for an identifier.
func (r *renamer) docComment(pkg Package, id *ast.Ident) *ast.CommentGroup {
	_, tokFile, nodes, _ := pathEnclosingInterval(r.ctx, r.snapshot, pkg, id.Pos(), id.End())
	for _, node := range nodes {
		switch decl := node.(type) {
		case *ast.FuncDecl:
			return decl.Doc
		case *ast.Field:
			return decl.Doc
		case *ast.GenDecl:
			return decl.Doc
		// For {Type,Value}Spec, if the doc on the spec is absent,
		// search for the enclosing GenDecl
		case *ast.TypeSpec:
			if decl.Doc != nil {
				return decl.Doc
			}
		case *ast.ValueSpec:
			if decl.Doc != nil {
				return decl.Doc
			}
		case *ast.Ident:
		case *ast.AssignStmt:
			// *ast.AssignStmt doesn't have an associated comment group.
			// So, we try to find a comment just before the identifier.

			// Try to find a comment group only for short variable declarations (:=).
			if decl.Tok != token.DEFINE {
				return nil
			}

			identLine := tokFile.Line(id.Pos())
			for _, comment := range nodes[len(nodes)-1].(*ast.File).Comments {
				if comment.Pos() > id.Pos() {
					// Comment is after the identifier.
					continue
				}

				lastCommentLine := tokFile.Line(comment.End())
				if lastCommentLine+1 == identLine {
					return comment
				}
			}
		default:
			return nil
		}
	}
	return nil
}

// updatePkgName returns the updates to rename a pkgName in the import spec by
// only modifying the package name portion of the import declaration.
func (r *renamer) updatePkgName(pkgName *types.PkgName) (*diff.Edit, error) {
	// Modify ImportSpec syntax to add or remove the Name as needed.
	pkg := r.packages[pkgName.Pkg()]
	_, tokFile, path, _ := pathEnclosingInterval(r.ctx, r.snapshot, pkg, pkgName.Pos(), pkgName.Pos())
	if len(path) < 2 {
		return nil, fmt.Errorf("no path enclosing interval for %s", pkgName.Name())
	}
	spec, ok := path[1].(*ast.ImportSpec)
	if !ok {
		return nil, fmt.Errorf("failed to update PkgName for %s", pkgName.Name())
	}

	newText := ""
	if pkgName.Imported().Name() != r.to {
		newText = r.to + " "
	}

	// Replace the portion (possibly empty) of the spec before the path:
	//     local "path"      or      "path"
	//   ->      <-                -><-
	start, end, err := safetoken.Offsets(tokFile, spec.Pos(), spec.Path.Pos())
	if err != nil {
		return nil, err
	}

	return &diff.Edit{
		Start: start,
		End:   end,
		New:   newText,
	}, nil
}

// qualifiedObjsAtProtocolPos returns info for all the types.Objects referenced
// at the given position, for the following selection of packages:
//
// 1. all packages (including all test variants), in their workspace parse mode
// 2. if not included above, at least one package containing uri in full parse mode
//
// Finding objects in (1) ensures that we locate references within all
// workspace packages, including in x_test packages. Including (2) ensures that
// we find local references in the current package, for non-workspace packages
// that may be open.
func qualifiedObjsAtProtocolPos(ctx context.Context, s Snapshot, uri span.URI, pp protocol.Position) ([]qualifiedObject, error) {
	fh, err := s.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	content, err := fh.Read()
	if err != nil {
		return nil, err
	}
	m := protocol.NewMapper(uri, content)
	offset, err := m.PositionOffset(pp)
	if err != nil {
		return nil, err
	}
	return qualifiedObjsAtLocation(ctx, s, positionKey{uri, offset}, map[positionKey]bool{})
}

// A qualifiedObject is the result of resolving a reference from an
// identifier to an object.
type qualifiedObject struct {
	obj types.Object // the referenced object
	pkg Package      // the Package that defines the object (nil => universe)
}

// A positionKey identifies a byte offset within a file (URI).
//
// When a file has been parsed multiple times in the same FileSet,
// there may be multiple token.Pos values denoting the same logical
// position. In such situations, a positionKey may be used for
// de-duplication.
type positionKey struct {
	uri    span.URI
	offset int
}

// qualifiedObjsAtLocation finds all objects referenced at offset in uri,
// across all packages in the snapshot.
func qualifiedObjsAtLocation(ctx context.Context, s Snapshot, key positionKey, seen map[positionKey]bool) ([]qualifiedObject, error) {
	if seen[key] {
		return nil, nil
	}
	seen[key] = true

	// We search for referenced objects starting with all packages containing the
	// current location, and then repeating the search for every distinct object
	// location discovered.
	//
	// In the common case, there should be at most one additional location to
	// consider: the definition of the object referenced by the location. But we
	// try to be comprehensive in case we ever support variations on build
	// constraints.
	metas, err := s.MetadataForFile(ctx, key.uri)
	if err != nil {
		return nil, err
	}
	ids := make([]PackageID, len(metas))
	for i, m := range metas {
		ids[i] = m.ID
	}
	pkgs, err := s.TypeCheck(ctx, TypecheckWorkspace, ids...)
	if err != nil {
		return nil, err
	}

	// In order to allow basic references/rename/implementations to function when
	// non-workspace packages are open, ensure that we have at least one fully
	// parsed package for the current file. This allows us to find references
	// inside the open package. Use WidestPackage to capture references in test
	// files.
	hasFullPackage := false
	for _, pkg := range pkgs {
		if pkg.ParseMode() == ParseFull {
			hasFullPackage = true
			break
		}
	}
	if !hasFullPackage {
		pkg, _, err := PackageForFile(ctx, s, key.uri, TypecheckFull, WidestPackage)
		if err != nil {
			return nil, err
		}
		pkgs = append(pkgs, pkg)
	}

	// report objects in the order we encounter them. This ensures that the first
	// result is at the cursor...
	var qualifiedObjs []qualifiedObject
	// ...but avoid duplicates.
	seenObjs := map[types.Object]bool{}

	for _, searchpkg := range pkgs {
		pgf, err := searchpkg.File(key.uri)
		if err != nil {
			return nil, err
		}
		pos := pgf.Tok.Pos(key.offset)

		// TODO(adonovan): replace this section with a call to objectsAt().
		path := pathEnclosingObjNode(pgf.File, pos)
		if path == nil {
			continue
		}
		var objs []types.Object
		switch leaf := path[0].(type) {
		case *ast.Ident:
			// If leaf represents an implicit type switch object or the type
			// switch "assign" variable, expand to all of the type switch's
			// implicit objects.
			if implicits, _ := typeSwitchImplicits(searchpkg.GetTypesInfo(), path); len(implicits) > 0 {
				objs = append(objs, implicits...)
			} else {
				obj := searchpkg.GetTypesInfo().ObjectOf(leaf)
				if obj == nil {
					return nil, fmt.Errorf("no object found for %q", leaf.Name)
				}
				objs = append(objs, obj)
			}
		case *ast.ImportSpec:
			// Look up the implicit *types.PkgName.
			obj := searchpkg.GetTypesInfo().Implicits[leaf]
			if obj == nil {
				return nil, fmt.Errorf("no object found for import %s", UnquoteImportPath(leaf))
			}
			objs = append(objs, obj)
		}

		// Get all of the transitive dependencies of the search package.
		pkgSet := map[*types.Package]Package{
			searchpkg.GetTypes(): searchpkg,
		}
		deps := recursiveDeps(s, searchpkg.Metadata())[1:]
		// Ignore the error from type checking, but check if the context was
		// canceled (which would have caused TypeCheck to exit early).
		depPkgs, _ := s.TypeCheck(ctx, TypecheckWorkspace, deps...)
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		for _, dep := range depPkgs {
			// Since we ignored the error from type checking, pkg may be nil.
			if dep != nil {
				pkgSet[dep.GetTypes()] = dep
			}
		}

		for _, obj := range objs {
			if obj.Parent() == types.Universe {
				return nil, fmt.Errorf("%q: builtin object", obj.Name())
			}
			pkg, ok := pkgSet[obj.Pkg()]
			if !ok {
				event.Error(ctx, fmt.Sprintf("no package for obj %s: %v", obj, obj.Pkg()), err)
				continue
			}
			qualifiedObjs = append(qualifiedObjs, qualifiedObject{obj: obj, pkg: pkg})
			seenObjs[obj] = true

			// If the qualified object is in another file (or more likely, another
			// package), it's possible that there is another copy of it in a package
			// that we haven't searched, e.g. a test variant. See golang/go#47564.
			//
			// In order to be sure we've considered all packages, call
			// qualifiedObjsAtLocation recursively for all locations we encounter. We
			// could probably be more precise here, only continuing the search if obj
			// is in another package, but this should be good enough to find all
			// uses.

			if key, found := packagePositionKey(pkg, obj.Pos()); found {
				otherObjs, err := qualifiedObjsAtLocation(ctx, s, key, seen)
				if err != nil {
					return nil, err
				}
				for _, other := range otherObjs {
					if !seenObjs[other.obj] {
						qualifiedObjs = append(qualifiedObjs, other)
						seenObjs[other.obj] = true
					}
				}
			} else {
				return nil, fmt.Errorf("missing file for position of %q in %q", obj.Name(), obj.Pkg().Name())
			}
		}
	}
	// Return an error if no objects were found since callers will assume that
	// the slice has at least 1 element.
	if len(qualifiedObjs) == 0 {
		return nil, errNoObjectFound
	}
	return qualifiedObjs, nil
}

// packagePositionKey finds the positionKey for the given pos.
//
// The second result reports whether the position was found.
func packagePositionKey(pkg Package, pos token.Pos) (positionKey, bool) {
	for _, pgf := range pkg.CompiledGoFiles() {
		offset, err := safetoken.Offset(pgf.Tok, pos)
		if err == nil {
			return positionKey{pgf.URI, offset}, true
		}
	}
	return positionKey{}, false
}

// ReferenceInfo holds information about reference to an identifier in Go source.
type ReferenceInfo struct {
	MappedRange   protocol.MappedRange
	ident         *ast.Ident
	obj           types.Object
	pkg           Package
	isDeclaration bool
}

// references is a helper function to avoid recomputing qualifiedObjsAtProtocolPos.
// The first element of qos is considered to be the declaration;
// if isDeclaration, the first result is an extra item for it.
// Only the definition-related fields of qualifiedObject are used.
// (Arguably it should accept a smaller data type.)
//
// This implementation serves Server.rename. TODO(adonovan): obviate it.
func references(ctx context.Context, snapshot Snapshot, qos []qualifiedObject) ([]*ReferenceInfo, error) {
	var (
		references []*ReferenceInfo
		seen       = make(map[positionKey]bool)
	)

	pos := qos[0].obj.Pos()
	if pos == token.NoPos {
		return nil, fmt.Errorf("no position for %s", qos[0].obj) // e.g. error.Error
	}
	// Inv: qos[0].pkg != nil, since Pos is valid.
	// Inv: qos[*].pkg != nil, since all qos are logically the same declaration.
	filename := safetoken.StartPosition(qos[0].pkg.FileSet(), pos).Filename
	pgf, err := qos[0].pkg.File(span.URIFromPath(filename))
	if err != nil {
		return nil, err
	}
	declIdent, err := findIdentifier(ctx, snapshot, qos[0].pkg, pgf, qos[0].obj.Pos())
	if err != nil {
		return nil, err
	}
	// Make sure declaration is the first item in the response.
	references = append(references, &ReferenceInfo{
		MappedRange:   declIdent.MappedRange,
		ident:         declIdent.ident,
		obj:           qos[0].obj,
		pkg:           declIdent.pkg,
		isDeclaration: true,
	})

	for _, qo := range qos {
		var searchPkgs []Package

		// Only search dependents if the object is exported.
		if qo.obj.Exported() {
			// If obj is a package-level object, we need only search
			// among direct reverse dependencies.
			// TODO(adonovan): opt: this will still spuriously search
			// transitively for (e.g.) capitalized local variables.
			// We could do better by checking for an objectpath.
			transitive := qo.obj.Pkg().Scope().Lookup(qo.obj.Name()) != qo.obj
			rdeps, err := snapshot.ReverseDependencies(ctx, qo.pkg.Metadata().ID, transitive)
			if err != nil {
				return nil, err
			}
			ids := make([]PackageID, 0, len(rdeps))
			for _, rdep := range rdeps {
				ids = append(ids, rdep.ID)
			}
			// TODO(adonovan): opt: build a search index
			// that doesn't require type checking.
			reverseDeps, err := snapshot.TypeCheck(ctx, TypecheckFull, ids...)
			if err != nil {
				return nil, err
			}
			searchPkgs = append(searchPkgs, reverseDeps...)
		}
		// Add the package in which the identifier is declared.
		searchPkgs = append(searchPkgs, qo.pkg)
		for _, pkg := range searchPkgs {
			for ident, obj := range pkg.GetTypesInfo().Uses {
				// For instantiated objects (as in methods or fields on instantiated
				// types), we may not have pointer-identical objects but still want to
				// consider them references.
				if !equalOrigin(obj, qo.obj) {
					// If ident is not a use of qo.obj, skip it, with one exception:
					// uses of an embedded field can be considered references of the
					// embedded type name
					v, ok := obj.(*types.Var)
					if !ok || !v.Embedded() {
						continue
					}
					named, ok := v.Type().(*types.Named)
					if !ok || named.Obj() != qo.obj {
						continue
					}
				}
				key, found := packagePositionKey(pkg, ident.Pos())
				if !found {
					bug.Reportf("ident %v (pos: %v) not found in package %v", ident.Name, ident.Pos(), pkg.Metadata().ID)
					continue
				}
				if seen[key] {
					continue
				}
				seen[key] = true
				filename := pkg.FileSet().File(ident.Pos()).Name()
				pgf, err := pkg.File(span.URIFromPath(filename))
				if err != nil {
					return nil, err
				}
				rng, err := pgf.NodeMappedRange(ident)
				if err != nil {
					return nil, err
				}
				references = append(references, &ReferenceInfo{
					ident:       ident,
					pkg:         pkg,
					obj:         obj,
					MappedRange: rng,
				})
			}
		}
	}

	return references, nil
}

// equalOrigin reports whether obj1 and obj2 have equivalent origin object.
// This may be the case even if obj1 != obj2, if one or both of them is
// instantiated.
func equalOrigin(obj1, obj2 types.Object) bool {
	return obj1.Pkg() == obj2.Pkg() && obj1.Pos() == obj2.Pos() && obj1.Name() == obj2.Name()
}

// parsePackageNameDecl is a convenience function that parses and
// returns the package name declaration of file fh, and reports
// whether the position ppos lies within it.
//
// Note: also used by references2.
func parsePackageNameDecl(ctx context.Context, snapshot Snapshot, fh FileHandle, ppos protocol.Position) (*ParsedGoFile, bool, error) {
	pgf, err := snapshot.ParseGo(ctx, fh, ParseHeader)
	if err != nil {
		return nil, false, err
	}
	// Careful: because we used ParseHeader,
	// pgf.Pos(ppos) may be beyond EOF => (0, err).
	pos, _ := pgf.PositionPos(ppos)
	return pgf, pgf.File.Name.Pos() <= pos && pos <= pgf.File.Name.End(), nil
}
