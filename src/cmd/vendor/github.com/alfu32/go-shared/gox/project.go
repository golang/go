package gox

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"go/build"
	"go/parser"
	"go/token"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

const Version = "0.1.0-dev.2"

type ProjectOptions struct {
	Options
	Context      build.Context
	CacheDir     string
	ExcludeTests bool
}
type Overlay struct{ Replace map[string]string }

// GeneratedName preserves platform and _test suffixes, allowing stock discovery.
func GeneratedName(name string) string { return strings.TrimSuffix(name, ".gox") + ".go" }
func Selected(path string, src []byte, ctx build.Context) (bool, error) {
	if ctx.GOOS == "" {
		ctx = build.Default
	}
	ctx.OpenFile = func(string) (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(src)), nil }
	name := filepath.Base(path)
	if strings.HasSuffix(name, ".gox") {
		name = GeneratedName(name)
	}
	return ctx.MatchFile(filepath.Dir(path), name)
}
func TransformCached(path string, src []byte, opts ProjectOptions) (string, error) {
	if opts.HookBindings == nil {
		files := map[string][]byte{}
		entries, e := os.ReadDir(filepath.Dir(path))
		if e != nil {
			return "", e
		}
		for _, entry := range entries {
			name := entry.Name()
			if opts.ExcludeTests && (strings.HasSuffix(name, "_test.go") || strings.HasSuffix(name, "_test.gox")) {
				continue
			}
			if entry.IsDir() || (!strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, ".gox")) {
				continue
			}
			p := filepath.Join(filepath.Dir(path), name)
			b, e := os.ReadFile(p)
			if e != nil {
				return "", e
			}
			if p == path {
				b = src
			}
			selected, e := Selected(p, b, opts.Context)
			if e != nil {
				return "", e
			}
			if selected {
				files[p] = b
			}
		}
		bindings, e := PackageHookBindings(files)
		if e != nil {
			return "", e
		}
		f, e := parser.ParseFile(token.NewFileSet(), path, src, parser.PackageClauseOnly)
		if e != nil {
			return "", e
		}
		opts.HookBindings = append([]string{}, bindings[f.Name.Name]...)
	}

	if opts.CacheDir == "" {
		d, e := os.UserCacheDir()
		if e != nil {
			return "", e
		}
		opts.CacheDir = filepath.Join(d, "gox")
	}
	if opts.Identity == "" {
		opts.Identity = SourceIdentity(path)
	}
	opts.SourcePositions = true
	fingerprint, e := executableFingerprint()
	if e != nil {
		return "", e
	}
	config, _ := json.Marshal(opts.Options)
	sum := sha256.Sum256(append([]byte(Version+"\x00"+fingerprint+"\x00"+path+"\x00"+string(config)+"\x00"), src...))
	name := filepath.Join(opts.CacheDir, hex.EncodeToString(sum[:])+".go")
	if _, e := os.Stat(name); e == nil {
		return name, nil
	}
	r, e := TransformFile(path, src, opts.Options)
	if e != nil {
		return "", e
	}
	if e = os.MkdirAll(opts.CacheDir, 0700); e != nil {
		// Some build sandboxes deliberately provide an unwritable HOME.
		// Keep normal user caches preferred, but retain a usable private temp cache.
		opts.CacheDir = filepath.Join(os.TempDir(), "gox-cache")
		name = filepath.Join(opts.CacheDir, hex.EncodeToString(sum[:])+".go")
		if e = os.MkdirAll(opts.CacheDir, 0700); e != nil {
			return "", e
		}
		if _, e = os.Stat(name); e == nil {
			return name, nil
		}
	}
	f, e := os.CreateTemp(opts.CacheDir, ".pending-")
	if e != nil {
		return "", e
	}
	temp := f.Name()
	defer os.Remove(temp)
	if _, e = f.Write(r.Source); e != nil {
		f.Close()
		return "", e
	}
	if e = f.Close(); e != nil {
		return "", e
	}
	if e = os.Rename(temp, name); e != nil {
		if _, exists := os.Stat(name); exists == nil {
			return name, nil
		}
		return "", e
	}
	return name, nil
}

// PrepareOverlay leaves the module layout and assets in place. Only GoX source
// paths gain virtual .go siblings; no source files or module manifests are edited.
func PrepareOverlay(root string, opts ProjectOptions) (Overlay, error) {
	out := Overlay{Replace: map[string]string{}}
	root, e := filepath.Abs(root)
	if e != nil {
		return out, e
	}
	e = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if path != root && (strings.HasPrefix(d.Name(), ".") || d.Name() == "vendor" || d.Name() == "node_modules") {
				return filepath.SkipDir
			}
			if path != root {
				if _, e := os.Stat(filepath.Join(path, "go.mod")); e == nil {
					return filepath.SkipDir
				}
			}
			return nil
		}
		if !strings.HasSuffix(path, ".gox") || opts.ExcludeTests && strings.HasSuffix(path, "_test.gox") {
			return nil
		}
		src, e := os.ReadFile(path)
		if e != nil {
			return e
		}
		ok, e := Selected(path, src, opts.Context)
		if e != nil {
			return e
		}
		if !ok {
			return nil
		}
		target := GeneratedName(path)
		if _, e = os.Stat(target); e == nil {
			return fmt.Errorf("GoX generated path conflicts with existing file: %s", target)
		}
		generated, e := TransformCached(path, src, opts)
		if e != nil {
			return e
		}
		out.Replace[target] = generated
		return nil
	})
	return out, e
}
func (o Overlay) Write(path string) error {
	b, e := json.Marshal(o)
	if e != nil {
		return e
	}
	return os.WriteFile(path, b, 0600)
}

// SourceIdentity gives component definitions a module-relative identity that is
// stable across checkouts. Diagnostics continue to use the real source filename.
func SourceIdentity(path string) string {
	path, _ = filepath.Abs(path)
	for dir := filepath.Dir(path); ; dir = filepath.Dir(dir) {
		if b, e := os.ReadFile(filepath.Join(dir, "go.mod")); e == nil {
			for _, line := range strings.Split(string(b), "\n") {
				fields := strings.Fields(line)
				if len(fields) >= 2 && fields[0] == "module" {
					rel, _ := filepath.Rel(dir, path)
					return strings.Trim(fields[1], "\"") + "/" + filepath.ToSlash(rel)
				}
			}
		}
		if filepath.Dir(dir) == dir {
			break
		}
	}
	return filepath.Base(path)
}
