// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fipsdit provides an analyzer which checks that exported functions in
// crypto/... do not call into crypto/internal/fips140/... or constant-time
// scalar primitives (crypto/subtle.ConstantTime{Select,ByteEq,Eq,LessOrEq} and
// crypto/internal/constanttime) without enabling Data Independent Timing via
// crypto/subtle.WithDataIndependentTiming.
package fipsdit

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"slices"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
)

// allowlist contains exported functions and FIPS targets that do not
// require constant-time execution (DIT).
var allowlist = map[string]bool{
	"crypto/internal/fips140/aes.Block.BlockSize":        true,
	"crypto/internal/fips140/aes.CBCDecrypter.BlockSize": true,
	"crypto/internal/fips140/aes.CBCDecrypter.SetIV":     true,
	"crypto/internal/fips140/aes.CBCEncrypter.BlockSize": true,
	"crypto/internal/fips140/aes.CBCEncrypter.SetIV":     true,
	"crypto/internal/fips140/aes.NewCBCDecrypter":        true,
	"crypto/internal/fips140/aes.NewCBCEncrypter":        true,
	"crypto/internal/fips140/aes.NewCTR":                 true,

	"crypto/internal/fips140/aes/gcm.GCM.NonceSize":                    true,
	"crypto/internal/fips140/aes/gcm.GCM.Overhead":                     true,
	"crypto/internal/fips140/aes/gcm.GCMForTLS12.NonceSize":            true,
	"crypto/internal/fips140/aes/gcm.GCMForTLS12.Overhead":             true,
	"crypto/internal/fips140/aes/gcm.GCMForTLS13.NonceSize":            true,
	"crypto/internal/fips140/aes/gcm.GCMForTLS13.Overhead":             true,
	"crypto/internal/fips140/aes/gcm.GCMWithCounterNonce.NonceSize":    true,
	"crypto/internal/fips140/aes/gcm.GCMWithCounterNonce.Overhead":     true,
	"crypto/internal/fips140/aes/gcm.GCMWithXORCounterNonce.NonceSize": true,
	"crypto/internal/fips140/aes/gcm.GCMWithXORCounterNonce.Overhead":  true,

	"crypto/internal/fips140/alias.AnyOverlap":     true,
	"crypto/internal/fips140/alias.InexactOverlap": true,

	// NewModulus is only called with public RSA moduli (via fipsPublicKey).
	"crypto/internal/fips140/bigmod.NewModulus": true,

	"crypto/internal/fips140/drbg.IsDefaultReader": true,

	"crypto/internal/fips140/ecdh.NewPublicKey":         true,
	"crypto/internal/fips140/ecdh.P256":                 true,
	"crypto/internal/fips140/ecdh.P384":                 true,
	"crypto/internal/fips140/ecdh.P521":                 true,
	"crypto/internal/fips140/ecdh.PrivateKey.Bytes":     true,
	"crypto/internal/fips140/ecdh.PrivateKey.PublicKey": true,
	"crypto/internal/fips140/ecdh.PublicKey.Bytes":      true,

	"crypto/internal/fips140/ecdsa.NewPublicKey":         true,
	"crypto/internal/fips140/ecdsa.P224":                 true,
	"crypto/internal/fips140/ecdsa.P256":                 true,
	"crypto/internal/fips140/ecdsa.P384":                 true,
	"crypto/internal/fips140/ecdsa.P521":                 true,
	"crypto/internal/fips140/ecdsa.PrivateKey.Bytes":     true,
	"crypto/internal/fips140/ecdsa.PrivateKey.PublicKey": true,
	"crypto/internal/fips140/ecdsa.PublicKey.Bytes":      true,
	"crypto/internal/fips140/ecdsa.Verify":               true,

	"crypto/internal/fips140/ed25519.NewPublicKey":         true,
	"crypto/internal/fips140/ed25519.PrivateKey.Bytes":     true,
	"crypto/internal/fips140/ed25519.PrivateKey.PublicKey": true,
	"crypto/internal/fips140/ed25519.Verify":               true,
	"crypto/internal/fips140/ed25519.VerifyCtx":            true,
	"crypto/internal/fips140/ed25519.VerifyPH":             true,

	"crypto/internal/fips140/hmac.HMAC.BlockSize": true,
	"crypto/internal/fips140/hmac.HMAC.Size":      true,

	"crypto/internal/fips140/mldsa.NewPublicKey44":       true,
	"crypto/internal/fips140/mldsa.NewPublicKey65":       true,
	"crypto/internal/fips140/mldsa.NewPublicKey87":       true,
	"crypto/internal/fips140/mldsa.PrivateKey.Bytes":     true,
	"crypto/internal/fips140/mldsa.PrivateKey.PublicKey": true,
	"crypto/internal/fips140/mldsa.PublicKey.Bytes":      true,
	"crypto/internal/fips140/mldsa.PublicKey.Equal":      true,
	"crypto/internal/fips140/mldsa.PublicKey.Parameters": true,
	"crypto/internal/fips140/mldsa.Verify":               true,

	"crypto/internal/fips140/mlkem.DecapsulationKey1024.Bytes":               true,
	"crypto/internal/fips140/mlkem.EncapsulationKey1024.EncapsulateInternal": true,
	"crypto/internal/fips140/mlkem.DecapsulationKey1024.EncapsulationKey":    true,
	"crypto/internal/fips140/mlkem.DecapsulationKey768.Bytes":                true,
	"crypto/internal/fips140/mlkem.EncapsulationKey768.EncapsulateInternal":  true,
	"crypto/internal/fips140/mlkem.DecapsulationKey768.EncapsulationKey":     true,
	"crypto/internal/fips140/mlkem.EncapsulationKey1024.Bytes":               true,
	"crypto/internal/fips140/mlkem.EncapsulationKey768.Bytes":                true,
	"crypto/internal/fips140/mlkem.NewEncapsulationKey768":                   true,
	"crypto/internal/fips140/mlkem.NewEncapsulationKey1024":                  true,

	"crypto/internal/fips140/nistec.P224Point.Bytes":    true,
	"crypto/internal/fips140/nistec.P224Point.SetBytes": true,
	"crypto/internal/fips140/nistec.P256Point.Bytes":    true,
	"crypto/internal/fips140/nistec.P256Point.SetBytes": true,
	"crypto/internal/fips140/nistec.P384Point.Bytes":    true,
	"crypto/internal/fips140/nistec.P384Point.SetBytes": true,
	"crypto/internal/fips140/nistec.P521Point.Bytes":    true,
	"crypto/internal/fips140/nistec.P521Point.SetBytes": true,

	"crypto/internal/fips140/rsa.GenerateKey":             true,
	"crypto/internal/fips140/rsa.PSSMaxSaltLength":        true,
	"crypto/internal/fips140/rsa.PrivateKey.Export":       true,
	"crypto/internal/fips140/rsa.PrivateKey.PublicKey":    true,
	"crypto/internal/fips140/rsa.VerifyPKCS1v15":          true,
	"crypto/internal/fips140/rsa.VerifyPSS":               true,
	"crypto/internal/fips140/rsa.VerifyPSSWithSaltLength": true,

	"crypto/internal/fips140/sha256.New":    true,
	"crypto/internal/fips140/sha256.New224": true,

	"crypto/internal/fips140/sha3.Digest.AppendBinary":    true,
	"crypto/internal/fips140/sha3.Digest.BlockSize":       true,
	"crypto/internal/fips140/sha3.Digest.MarshalBinary":   true,
	"crypto/internal/fips140/sha3.Digest.Reset":           true,
	"crypto/internal/fips140/sha3.Digest.Size":            true,
	"crypto/internal/fips140/sha3.Digest.UnmarshalBinary": true,
	"crypto/internal/fips140/sha3.New":                    true,
	"crypto/internal/fips140/sha3.New224":                 true,
	"crypto/internal/fips140/sha3.New256":                 true,
	"crypto/internal/fips140/sha3.New384":                 true,
	"crypto/internal/fips140/sha3.New512":                 true,
	"crypto/internal/fips140/sha3.NewCShake128":           true,
	"crypto/internal/fips140/sha3.NewCShake256":           true,
	"crypto/internal/fips140/sha3.NewShake128":            true,
	"crypto/internal/fips140/sha3.NewShake256":            true,
	"crypto/internal/fips140/sha3.SHAKE.AppendBinary":     true,
	"crypto/internal/fips140/sha3.SHAKE.BlockSize":        true,
	"crypto/internal/fips140/sha3.SHAKE.MarshalBinary":    true,
	"crypto/internal/fips140/sha3.SHAKE.Reset":            true,
	"crypto/internal/fips140/sha3.SHAKE.Size":             true,
	"crypto/internal/fips140/sha3.SHAKE.UnmarshalBinary":  true,

	"crypto/internal/fips140/sha512.New":        true,
	"crypto/internal/fips140/sha512.New384":     true,
	"crypto/internal/fips140/sha512.New512_224": true,
	"crypto/internal/fips140/sha512.New512_256": true,

	// Check only hashes public certificate metadata (issuer SPKI and serial).
	"crypto/internal/upki.Check": true,
}

// funcFact is an analysis.Fact exported on *types.Func objects.
type funcFact struct {
	// UnprotectedPath is non-empty if calling this function executes a path
	// to crypto/internal/fips140/... without subtle.WithDataIndependentTiming.
	UnprotectedPath string

	// ReturnsFIPSType is non-empty if this function returns a concrete type
	// defined in crypto/internal/fips140/... (e.g. "crypto/internal/fips140/sha256.Digest").
	ReturnsFIPSType string
}

func (*funcFact) AFact() {}

var Analyzer = &analysis.Analyzer{
	Name:      "fipsditcheck",
	Doc:       "check FIPS 140 Data Independent Timing (DIT) requirements",
	Requires:  []*analysis.Analyzer{inspect.Analyzer},
	FactTypes: []analysis.Fact{(*funcFact)(nil)},
	Run:       run,
}

func run(pass *analysis.Pass) (any, error) {
	if isFIPS140Pkg(pass.Pkg) || pass.Pkg.Path() == "crypto/internal/constanttime" || !strings.HasPrefix(pass.Pkg.Path(), "crypto") {
		return nil, nil
	}

	isAllowlisted := func(name string) bool {
		if name == "" {
			return false
		}
		return allowlist[name] || allowlist[normalizeFIPSPath(name)]
	}

	// Collect non-test function declarations and package-level variable
	// initializer closures in this package.
	type declInfo struct {
		decl *ast.FuncDecl
		fn   *types.Func
	}
	type varLitInfo struct {
		name string
		lit  *ast.FuncLit
	}
	var decls []declInfo
	var pkgFuncs []*types.Func
	var pkgVarLits []varLitInfo
	declMap := make(map[*types.Func]*ast.FuncDecl)

	for _, file := range pass.Files {
		filename := pass.Fset.File(file.Pos()).Name()
		if strings.HasSuffix(filename, "_test.go") {
			continue
		}
		for _, d := range file.Decls {
			switch decl := d.(type) {
			case *ast.FuncDecl:
				if decl.Name == nil {
					continue
				}
				obj, ok := pass.TypesInfo.Defs[decl.Name].(*types.Func)
				if !ok || obj == nil {
					continue
				}
				obj = obj.Origin()
				decls = append(decls, declInfo{decl: decl, fn: obj})
				pkgFuncs = append(pkgFuncs, obj)
				declMap[obj] = decl
			case *ast.GenDecl:
				if decl.Tok != token.VAR {
					continue
				}
				for _, spec := range decl.Specs {
					vs, ok := spec.(*ast.ValueSpec)
					if !ok {
						continue
					}
					for i, val := range vs.Values {
						varName := pass.Pkg.Path()
						if i < len(vs.Names) {
							varName = pass.Pkg.Path() + "." + vs.Names[i].Name
						}
						ast.Inspect(val, func(n ast.Node) bool {
							if call, ok := n.(*ast.CallExpr); ok && isWithDIT(pass.TypesInfo, call) {
								return false
							}
							if lit, ok := n.(*ast.FuncLit); ok {
								pkgVarLits = append(pkgVarLits, varLitInfo{name: varName, lit: lit})
								return false
							}
							return true
						})
					}
				}
			}
		}
	}

	// Identify functions that return a concrete crypto/internal/fips140/... type
	// (e.g. returning *sha256.Digest as a hash.Hash interface).
	returnsFIPS := make(map[*types.Func]string)
	for _, info := range decls {
		if info.decl.Type.Results != nil {
			for _, field := range info.decl.Type.Results.List {
				if t := pass.TypesInfo.TypeOf(field.Type); t != nil {
					if fipsType := extractFIPSTypeName(t); fipsType != "" {
						returnsFIPS[info.fn] = fipsType
						break
					}
				}
			}
		}
		if info.decl.Body == nil || returnsFIPS[info.fn] != "" {
			continue
		}
		ast.Inspect(info.decl.Body, func(n ast.Node) bool {
			if _, ok := n.(*ast.FuncLit); ok {
				return false
			}
			// Skip returns guarded by "if !sys.DITSupported { ... }" since the
			// unwrapped FIPS type is only returned when DIT is unavailable.
			if ifStmt, ok := n.(*ast.IfStmt); ok && isNotDITSupported(pass.TypesInfo, ifStmt.Cond) {
				if ifStmt.Else != nil {
					ast.Inspect(ifStmt.Else, func(en ast.Node) bool {
						if _, ok := en.(*ast.FuncLit); ok {
							return false
						}
						if ret, ok := en.(*ast.ReturnStmt); ok {
							for _, expr := range ret.Results {
								if t := pass.TypesInfo.TypeOf(expr); t != nil {
									if fipsType := extractFIPSTypeName(t); fipsType != "" {
										returnsFIPS[info.fn] = fipsType
										return false
									}
								}
							}
						}
						return true
					})
				}
				return false
			}
			ret, ok := n.(*ast.ReturnStmt)
			if !ok {
				return true
			}
			for _, expr := range ret.Results {
				if t := pass.TypesInfo.TypeOf(expr); t != nil {
					if fipsType := extractFIPSTypeName(t); fipsType != "" {
						returnsFIPS[info.fn] = fipsType
						return false
					}
				}
				if call, ok := stripParens(expr).(*ast.CallExpr); ok {
					if callee := calleeFunc(pass.TypesInfo, call); callee != nil {
						if isFIPS140Pkg(callee.Pkg()) {
							if sig, ok := callee.Type().(*types.Signature); ok && sig.Results().Len() > 0 {
								if fipsType := extractFIPSTypeName(sig.Results().At(0).Type()); fipsType != "" {
									returnsFIPS[info.fn] = fipsType
									return false
								}
							}
						} else if callee.Pkg() != pass.Pkg {
							var fact funcFact
							if pass.ImportObjectFact(callee, &fact) && fact.ReturnsFIPSType != "" {
								returnsFIPS[info.fn] = fact.ReturnsFIPSType
								return false
							}
						}
					}
				}
			}
			return true
		})
	}

	// Propagate returnsFIPS across intra-package wrapper returns.
	for range len(decls) {
		changed := false
		for _, info := range decls {
			if returnsFIPS[info.fn] != "" || info.decl.Body == nil {
				continue
			}
			ast.Inspect(info.decl.Body, func(n ast.Node) bool {
				if _, ok := n.(*ast.FuncLit); ok {
					return false
				}
				ret, ok := n.(*ast.ReturnStmt)
				if !ok {
					return true
				}
				for _, expr := range ret.Results {
					if call, ok := stripParens(expr).(*ast.CallExpr); ok {
						if callee := calleeFunc(pass.TypesInfo, call); callee != nil && callee.Pkg() == pass.Pkg {
							if ft := returnsFIPS[callee]; ft != "" {
								returnsFIPS[info.fn] = ft
								changed = true
								return false
							}
						}
					}
				}
				return true
			})
		}
		if !changed {
			break
		}
	}

	// Collect concrete FIPS types used to instantiate generic type parameters
	// in this package (e.g. nistCurve[*nistec.P256Point] in crypto/elliptic).
	// Key named-type parameters by (origin TypeName, index) because method
	// declarations create distinct *types.TypeParam objects for receiver type
	// parameters rather than reusing the type declaration's *types.TypeParam.
	type typeParamKey struct {
		owner types.Object
		index int
	}
	typeParamFIPSTypes := make(map[typeParamKey][]string)
	addTypeParamFIPS := func(key typeParamKey, ft string) {
		if !slices.Contains(typeParamFIPSTypes[key], ft) {
			typeParamFIPSTypes[key] = append(typeParamFIPSTypes[key], ft)
		}
	}
	for ident, inst := range pass.TypesInfo.Instances {
		var owner types.Object
		var tparams *types.TypeParamList
		switch obj := pass.TypesInfo.Uses[ident].(type) {
		case *types.TypeName:
			if named, ok := types.Unalias(obj.Type()).(*types.Named); ok {
				owner = named.Origin().Obj()
				tparams = named.TypeParams()
			}
		case *types.Func:
			if sig, ok := obj.Type().(*types.Signature); ok {
				owner = obj.Origin()
				tparams = sig.TypeParams()
			}
		}
		if owner == nil || tparams == nil {
			continue
		}
		for i := range min(tparams.Len(), inst.TypeArgs.Len()) {
			if ft := extractFIPSTypeName(inst.TypeArgs.At(i)); ft != "" {
				addTypeParamFIPS(typeParamKey{owner, i}, ft)
			}
		}
	}

	// Collect unprotected call edges for each function in the package.
	// First, pre-scan all function bodies to discover local variable FIPS types
	// and concrete FIPS types stored into interface-typed struct fields (e.g.
	// prefixNonceAEAD.aead in crypto/tls).
	fieldFIPSTypes := make(map[*types.Var][]string)
	addFieldFIPS := func(fv *types.Var, ft string) {
		if fv == nil || ft == "" {
			return
		}
		fv = fv.Origin()
		if !slices.Contains(fieldFIPSTypes[fv], ft) {
			fieldFIPSTypes[fv] = append(fieldFIPSTypes[fv], ft)
		}
	}

	walkers := make([]*unprotectedWalker, len(decls))
	for i, info := range decls {
		fn := info.fn
		if info.decl.Body == nil {
			continue
		}
		if isAllowlisted(formatFuncObj(fn)) || isAllowlisted(fn.FullName()) {
			continue
		}

		// Map this function's/method's own *types.TypeParam objects to the
		// concrete FIPS types collected for its owner.
		fnTypeParamFIPS := make(map[*types.TypeParam][]string)
		if sig, ok := fn.Type().(*types.Signature); ok {
			if recv := sig.Recv(); recv != nil {
				rt := types.Unalias(recv.Type())
				if ptr, ok := rt.(*types.Pointer); ok {
					rt = types.Unalias(ptr.Elem())
				}
				if named, ok := rt.(*types.Named); ok {
					owner := named.Origin().Obj()
					rtp := sig.RecvTypeParams()
					for j := range rtp.Len() {
						if fts := typeParamFIPSTypes[typeParamKey{owner, j}]; len(fts) > 0 {
							fnTypeParamFIPS[rtp.At(j)] = fts
						}
					}
				}
			}
			ftp := sig.TypeParams()
			for j := range ftp.Len() {
				if fts := typeParamFIPSTypes[typeParamKey{fn.Origin(), j}]; len(fts) > 0 {
					fnTypeParamFIPS[ftp.At(j)] = fts
				}
			}
		}

		w := &unprotectedWalker{
			pass:               pass,
			fn:                 fn,
			returnsFIPS:        returnsFIPS,
			typeParamFIPSTypes: fnTypeParamFIPS,
			fieldFIPSTypes:     fieldFIPSTypes,
			isAllowlisted:      isAllowlisted,
			varLits:            make(map[*types.Var]*ast.FuncLit),
			varFIPSType:        make(map[*types.Var]string),
			visitedLits:        make(map[*ast.FuncLit]bool),
		}
		w.preScan(info.decl.Body)
		w.scanFieldFIPSTypes(info.decl.Body, addFieldFIPS)
		walkers[i] = w
	}

	directTarget := make(map[*types.Func]string)
	localCallees := make(map[*types.Func][]*types.Func)

	for i, info := range decls {
		w := walkers[i]
		if w == nil {
			continue
		}
		ast.Walk(w, info.decl.Body)

		if w.firstDirect != "" {
			directTarget[info.fn] = w.firstDirect
		}
		localCallees[info.fn] = w.localCalls
	}

	// Compute shortest unprotected paths to crypto/internal/fips140/...
	dist := make(map[*types.Func]int)
	tailPath := make(map[*types.Func]string)

	for _, fn := range pkgFuncs {
		if target := directTarget[fn]; target != "" {
			dist[fn] = 1
			tailPath[fn] = target
		}
	}

	changed := true
	for changed {
		changed = false
		for _, u := range pkgFuncs {
			if isAllowlisted(formatFuncObj(u)) || isAllowlisted(u.FullName()) {
				continue
			}
			for _, v := range localCallees[u] {
				if isAllowlisted(formatFuncObj(v)) || isAllowlisted(v.FullName()) {
					continue
				}
				if d, ok := dist[v]; ok {
					if curD, hasCur := dist[u]; !hasCur || d+1 < curD {
						dist[u] = d + 1
						tailPath[u] = formatFuncObj(v) + " -> " + tailPath[v]
						changed = true
					}
				}
			}
		}
	}

	// Export facts and report diagnostics on exported functions/methods.
	reportPkg := !isInternalPkg(pass.Pkg.Path())
	for _, info := range decls {
		u := info.fn
		fact := &funcFact{
			ReturnsFIPSType: returnsFIPS[u],
		}
		var unprotectedPath string
		if d, ok := dist[u]; ok && d > 0 && !isAllowlisted(formatFuncObj(u)) && !isAllowlisted(u.FullName()) {
			unprotectedPath = formatFuncObj(u) + " -> " + tailPath[u]
		}
		// Export UnprotectedPath across package boundaries for internal helper packages
		// (or unexported linkname targets). Exported functions in public packages are
		// themselves the DIT boundary and are reported directly in their defining package.
		if isInternalPkg(pass.Pkg.Path()) || !u.Exported() {
			fact.UnprotectedPath = unprotectedPath
		}
		if fact.UnprotectedPath != "" || fact.ReturnsFIPSType != "" {
			pass.ExportObjectFact(u, fact)
		}

		if reportPkg && unprotectedPath != "" && u.Exported() {
			pass.Reportf(
				info.decl.Name.Pos(),
				"exported function %s calls constant-time/FIPS code without subtle.WithDataIndependentTiming: %s",
				formatFuncObj(u),
				unprotectedPath,
			)
		}
	}

	// Also check closures inside package-level variable initializers (e.g.
	// crypto/ecdh's p256/p384/p521 nistCurve function fields).
	if reportPkg {
		for _, vl := range pkgVarLits {
			w := &unprotectedWalker{
				pass:           pass,
				returnsFIPS:    returnsFIPS,
				fieldFIPSTypes: fieldFIPSTypes,
				isAllowlisted:  isAllowlisted,
				varLits:        make(map[*types.Var]*ast.FuncLit),
				varFIPSType:    make(map[*types.Var]string),
				visitedLits:    make(map[*ast.FuncLit]bool),
			}
			w.preScan(vl.lit.Body)
			ast.Walk(w, vl.lit.Body)
			path := w.firstDirect
			if path == "" {
				bestD := 0
				for _, v := range w.localCalls {
					if isAllowlisted(formatFuncObj(v)) || isAllowlisted(v.FullName()) {
						continue
					}
					if d, ok := dist[v]; ok && (bestD == 0 || d < bestD) {
						bestD = d
						path = formatFuncObj(v) + " -> " + tailPath[v]
					}
				}
			}
			if path != "" {
				pass.Reportf(
					vl.lit.Pos(),
					"package variable %s initializer calls constant-time/FIPS code without subtle.WithDataIndependentTiming: %s -> %s",
					vl.name,
					vl.name,
					path,
				)
			}
		}
	}

	return nil, nil
}

type unprotectedWalker struct {
	pass               *analysis.Pass
	fn                 *types.Func
	returnsFIPS        map[*types.Func]string
	typeParamFIPSTypes map[*types.TypeParam][]string
	fieldFIPSTypes     map[*types.Var][]string
	isAllowlisted      func(string) bool

	varLits     map[*types.Var]*ast.FuncLit
	varFIPSType map[*types.Var]string
	visitedLits map[*ast.FuncLit]bool

	firstDirect string
	localCalls  []*types.Func
}

func (w *unprotectedWalker) recordDirect(targetPath string) {
	if w.firstDirect == "" {
		w.firstDirect = targetPath
	}
}

func (w *unprotectedWalker) recordLocal(callee *types.Func) {
	if slices.Contains(w.localCalls, callee) {
		return
	}
	w.localCalls = append(w.localCalls, callee)
}

// preScan finds local variables assigned closures or FIPS-backed values.
func (w *unprotectedWalker) preScan(body *ast.BlockStmt) {
	ast.Inspect(body, func(n ast.Node) bool {
		switch stmt := n.(type) {
		case *ast.AssignStmt:
			for i, rhs := range stmt.Rhs {
				if i >= len(stmt.Lhs) {
					break
				}
				ident, ok := stripParens(stmt.Lhs[i]).(*ast.Ident)
				if !ok {
					continue
				}
				v := lookupVar(w.pass.TypesInfo, ident)
				if v == nil {
					continue
				}
				if lit, ok := stripParens(rhs).(*ast.FuncLit); ok {
					w.varLits[v] = lit
				}
				if ft := w.exprFIPSType(rhs); ft != "" {
					w.varFIPSType[v] = ft
				}
			}
		case *ast.ValueSpec:
			for i, val := range stmt.Values {
				if i >= len(stmt.Names) {
					break
				}
				v := lookupVar(w.pass.TypesInfo, stmt.Names[i])
				if v == nil {
					continue
				}
				if lit, ok := stripParens(val).(*ast.FuncLit); ok {
					w.varLits[v] = lit
				}
				if ft := w.exprFIPSType(val); ft != "" {
					w.varFIPSType[v] = ft
				}
			}
		}
		return true
	})
}

// scanFieldFIPSTypes records concrete FIPS types stored into struct fields via
// composite literals or field assignments (e.g. &prefixNonceAEAD{aead: aead}).
func (w *unprotectedWalker) scanFieldFIPSTypes(body *ast.BlockStmt, add func(*types.Var, string)) {
	resolve := func(expr ast.Expr) string {
		if ft := w.exprFIPSType(expr); ft != "" {
			return ft
		}
		if ident, ok := stripParens(expr).(*ast.Ident); ok {
			if v := lookupVar(w.pass.TypesInfo, ident); v != nil {
				return w.varFIPSType[v]
			}
		}
		return ""
	}
	ast.Inspect(body, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.CompositeLit:
			t := w.pass.TypesInfo.TypeOf(node)
			if t == nil {
				return true
			}
			t = types.Unalias(t)
			if ptr, ok := t.(*types.Pointer); ok {
				t = types.Unalias(ptr.Elem())
			}
			st, ok := t.Underlying().(*types.Struct)
			if !ok {
				return true
			}
			for i, elt := range node.Elts {
				if kv, ok := elt.(*ast.KeyValueExpr); ok {
					if ident, ok := stripParens(kv.Key).(*ast.Ident); ok {
						if fv, ok := w.pass.TypesInfo.Uses[ident].(*types.Var); ok {
							add(fv, resolve(kv.Value))
						}
					}
				} else if i < st.NumFields() {
					add(st.Field(i), resolve(elt))
				}
			}
		case *ast.AssignStmt:
			for i, lhs := range node.Lhs {
				if i >= len(node.Rhs) {
					break
				}
				sel, ok := stripParens(lhs).(*ast.SelectorExpr)
				if !ok {
					continue
				}
				if s, ok := w.pass.TypesInfo.Selections[sel]; ok {
					if fv, ok := s.Obj().(*types.Var); ok {
						add(fv, resolve(node.Rhs[i]))
					}
				}
			}
		}
		return true
	})
}

func (w *unprotectedWalker) exprFIPSType(expr ast.Expr) string {
	if t := w.pass.TypesInfo.TypeOf(expr); t != nil {
		if ft := extractFIPSTypeName(t); ft != "" {
			return ft
		}
		if tup, ok := t.(*types.Tuple); ok && tup.Len() > 0 {
			if ft := extractFIPSTypeName(tup.At(0).Type()); ft != "" {
				return ft
			}
		}
	}
	if call, ok := stripParens(expr).(*ast.CallExpr); ok {
		if callee := calleeFunc(w.pass.TypesInfo, call); callee != nil {
			if ft := w.returnsFIPS[callee]; ft != "" {
				return ft
			}
			if callee.Pkg() != w.pass.Pkg {
				var fact funcFact
				if w.pass.ImportObjectFact(callee, &fact) && fact.ReturnsFIPSType != "" {
					return fact.ReturnsFIPSType
				}
			}
		}
	}
	return ""
}

func (w *unprotectedWalker) Visit(node ast.Node) ast.Visitor {
	if node == nil {
		return nil
	}

	switch n := node.(type) {
	case *ast.AssignStmt:
		for _, lhs := range n.Lhs {
			if ident, ok := stripParens(lhs).(*ast.Ident); ok && w.pass.TypesInfo.Defs[ident] != nil {
				continue
			}
			ast.Walk(w, lhs)
		}
		for i, rhs := range n.Rhs {
			if _, isLit := stripParens(rhs).(*ast.FuncLit); isLit && i < len(n.Lhs) {
				if ident, ok := stripParens(n.Lhs[i]).(*ast.Ident); ok {
					if v := lookupVar(w.pass.TypesInfo, ident); v != nil && w.varLits[v] != nil {
						continue
					}
				}
			}
			ast.Walk(w, rhs)
		}
		return nil

	case *ast.ValueSpec:
		if n.Type != nil {
			ast.Walk(w, n.Type)
		}
		for i, val := range n.Values {
			if _, isLit := stripParens(val).(*ast.FuncLit); isLit && i < len(n.Names) {
				if v := lookupVar(w.pass.TypesInfo, n.Names[i]); v != nil && w.varLits[v] != nil {
					continue
				}
			}
			ast.Walk(w, val)
		}
		return nil

	case *ast.CallExpr:
		if isWithDIT(w.pass.TypesInfo, n) {
			if len(n.Args) > 0 {
				arg := stripParens(n.Args[0])
				if !isFuncValueOrLit(arg) {
					ast.Walk(w, arg)
				}
			}
			return nil
		}

		// Check if calling a method on an interface variable, struct field, or
		// generic type parameter holding a concrete FIPS type.
		if sel, ok := stripParens(n.Fun).(*ast.SelectorExpr); ok {
			if xIdent, ok := stripParens(sel.X).(*ast.Ident); ok {
				if v := lookupVar(w.pass.TypesInfo, xIdent); v != nil {
					if fipsType := w.varFIPSType[v]; fipsType != "" {
						targetName := fipsType + "." + sel.Sel.Name
						if !w.isAllowlisted(targetName) {
							w.recordDirect(targetName)
						}
					}
				}
			}
			if xSel, ok := stripParens(sel.X).(*ast.SelectorExpr); ok {
				if fieldSel, ok := w.pass.TypesInfo.Selections[xSel]; ok {
					if fv, ok := fieldSel.Obj().(*types.Var); ok {
						for _, fipsType := range w.fieldFIPSTypes[fv.Origin()] {
							targetName := fipsType + "." + sel.Sel.Name
							if !w.isAllowlisted(targetName) {
								w.recordDirect(targetName)
							}
						}
					}
				}
			}
			if recvType := w.pass.TypesInfo.TypeOf(sel.X); recvType != nil {
				if tp, ok := types.Unalias(recvType).(*types.TypeParam); ok {
					for _, fipsType := range w.typeParamFIPSTypes[tp] {
						targetName := fipsType + "." + sel.Sel.Name
						if !w.isAllowlisted(targetName) {
							w.recordDirect(targetName)
						}
					}
				}
			}
		}

		if callee := calleeFunc(w.pass.TypesInfo, n); callee != nil {
			calleeName := formatFuncObj(callee)
			if !w.isAllowlisted(calleeName) && !w.isAllowlisted(callee.FullName()) {
				if isFIPS140Pkg(callee.Pkg()) || isSubtleIntrinsic(w.pass.Pkg, callee) {
					w.recordDirect(calleeName)
				} else if callee.Pkg() == w.pass.Pkg {
					w.recordLocal(callee)
				} else {
					var fact funcFact
					if w.pass.ImportObjectFact(callee, &fact) && fact.UnprotectedPath != "" {
						w.recordDirect(fact.UnprotectedPath)
					}
				}
			}
		}

		ast.Walk(w, n.Fun)
		for _, arg := range n.Args {
			ast.Walk(w, arg)
		}
		return nil

	case *ast.Ident:
		if v := lookupVar(w.pass.TypesInfo, n); v != nil {
			if lit, ok := w.varLits[v]; ok && !w.visitedLits[lit] {
				w.visitedLits[lit] = true
				ast.Walk(w, lit.Body)
			}
		}
		return nil

	case *ast.FuncLit:
		if !w.visitedLits[n] {
			w.visitedLits[n] = true
			ast.Walk(w, n.Body)
		}
		return nil
	}

	return w
}

func isWithDIT(info *types.Info, call *ast.CallExpr) bool {
	callee := calleeFunc(info, call)
	if callee == nil || callee.Pkg() == nil {
		return false
	}
	return callee.Pkg().Path() == "crypto/subtle" && callee.Name() == "WithDataIndependentTiming"
}

func isFuncValueOrLit(expr ast.Expr) bool {
	switch stripParens(expr).(type) {
	case *ast.FuncLit, *ast.Ident, *ast.SelectorExpr, *ast.IndexExpr, *ast.IndexListExpr:
		return true
	default:
		return false
	}
}

func calleeFunc(info *types.Info, call *ast.CallExpr) *types.Func {
	if call == nil {
		return nil
	}
	fun := stripParens(call.Fun)
	switch f := fun.(type) {
	case *ast.IndexExpr:
		fun = stripParens(f.X)
	case *ast.IndexListExpr:
		fun = stripParens(f.X)
	}

	switch f := fun.(type) {
	case *ast.Ident:
		if fn, ok := info.Uses[f].(*types.Func); ok {
			return fn.Origin()
		}
	case *ast.SelectorExpr:
		if sel, ok := info.Selections[f]; ok {
			if fn, ok := sel.Obj().(*types.Func); ok {
				return fn.Origin()
			}
		}
		if fn, ok := info.Uses[f.Sel].(*types.Func); ok {
			return fn.Origin()
		}
	}
	return nil
}

func lookupVar(info *types.Info, ident *ast.Ident) *types.Var {
	if ident == nil {
		return nil
	}
	if v, ok := info.Defs[ident].(*types.Var); ok {
		return v.Origin()
	}
	if v, ok := info.Uses[ident].(*types.Var); ok {
		return v.Origin()
	}
	return nil
}

func stripParens(expr ast.Expr) ast.Expr {
	for {
		p, ok := expr.(*ast.ParenExpr)
		if !ok {
			return expr
		}
		expr = p.X
	}
}

func isNotDITSupported(info *types.Info, cond ast.Expr) bool {
	u, ok := stripParens(cond).(*ast.UnaryExpr)
	if !ok || u.Op.String() != "!" {
		return false
	}
	sel, ok := stripParens(u.X).(*ast.SelectorExpr)
	if !ok || sel.Sel.Name != "DITSupported" {
		return false
	}
	obj := info.Uses[sel.Sel]
	return obj != nil && obj.Pkg() != nil && obj.Pkg().Path() == "internal/runtime/sys"
}

// normalizeFIPSPath strips any versioned FIPS 140-3 snapshot segment (such as
// "v1.26.0" or "v1.0.0-c2097c7c") immediately following "crypto/internal/fips140/".
func normalizeFIPSPath(path string) string {
	const prefix = "crypto/internal/fips140/"
	rest, ok := strings.CutPrefix(path, prefix)
	if !ok {
		return path
	}
	seg, after, hasSlash := strings.Cut(rest, "/")
	if len(seg) >= 2 && seg[0] == 'v' && seg[1] >= '0' && seg[1] <= '9' {
		if hasSlash {
			return prefix + after
		}
		return "crypto/internal/fips140"
	}
	return path
}

func isFIPS140Pkg(pkg *types.Package) bool {
	if pkg == nil {
		return false
	}
	return strings.HasPrefix(normalizeFIPSPath(pkg.Path()), "crypto/internal/fips140/")
}

func isSubtleIntrinsic(callerPkg *types.Package, callee *types.Func) bool {
	if callee == nil || callee.Pkg() == nil {
		return false
	}
	switch callee.Pkg().Path() {
	case "crypto/subtle":
		switch callee.Name() {
		case "ConstantTimeSelect", "ConstantTimeByteEq", "ConstantTimeEq", "ConstantTimeLessOrEq":
			return true
		}
	case "crypto/internal/constanttime":
		if callerPkg == nil || callerPkg.Path() != "crypto/subtle" {
			return true
		}
	}
	return false
}

func isInternalPkg(path string) bool {
	return path == "internal" || strings.HasPrefix(path, "internal/") || strings.Contains(path, "/internal/")
}

func extractFIPSTypeName(t types.Type) string {
	if t == nil {
		return ""
	}
	t = types.Unalias(t)
	if ptr, ok := t.(*types.Pointer); ok {
		t = types.Unalias(ptr.Elem())
	}
	if named, ok := t.(*types.Named); ok {
		obj := named.Obj()
		if obj != nil && isFIPS140Pkg(obj.Pkg()) {
			return normalizeFIPSPath(obj.Pkg().Path()) + "." + obj.Name()
		}
	}
	return ""
}

func formatFuncObj(fn *types.Func) string {
	if fn == nil {
		return ""
	}
	fn = fn.Origin()
	pkgPath := ""
	if fn.Pkg() != nil {
		pkgPath = normalizeFIPSPath(fn.Pkg().Path())
	}
	if sig, ok := fn.Type().(*types.Signature); ok && sig.Recv() != nil {
		recvType := types.Unalias(sig.Recv().Type())
		if ptr, ok := recvType.(*types.Pointer); ok {
			recvType = types.Unalias(ptr.Elem())
		}
		if named, ok := recvType.(*types.Named); ok && named.Obj() != nil {
			return fmt.Sprintf("%s.%s.%s", pkgPath, named.Obj().Name(), fn.Name())
		}
	}
	if pkgPath != "" {
		return fmt.Sprintf("%s.%s", pkgPath, fn.Name())
	}
	return fn.Name()
}
