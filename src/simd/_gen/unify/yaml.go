// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unify

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"
)

// ReadOpts provides options to [Read] and related functions. The zero value is
// the default options.
type ReadOpts struct {
	// FS, if non-nil, is the file system from which to resolve !import file
	// names.
	FS fs.FS
}

// Read reads a [Closure] in YAML format from r, using path for error messages.
//
// It maps YAML nodes into terminal Values as follows:
//
// - "_" or !top _ is the top value ([Top]).
//
// - "_|_" or !bottom _ is the bottom value. This is an error during
// unmarshaling, but can appear in marshaled values.
//
// - "$<name>" or !var <name> is a variable ([Var]). Everywhere the same name
// appears within a single unmarshal operation, it is mapped to the same
// variable. Different unmarshal operations get different variables, even if
// they have the same string name.
//
// - !regex "x" is a regular expression ([String]), as is any string that
// doesn't match "_", "_|_", or "$...". Regular expressions are implicitly
// anchored at the beginning and end. If the string doesn't contain any
// meta-characters (that is, it's a "literal" regular expression), then it's
// treated as an exact string.
//
// - !string "x", or any int, float, bool, or binary value is an exact string
// ([String]).
//
// - !regex [x, y, ...] is an intersection of regular expressions ([String]).
//
// It maps YAML nodes into non-terminal Values as follows:
//
// - Sequence nodes like [x, y, z] are tuples ([Tuple]).
//
// - !repeat [x] is a repeated tuple ([Tuple]), which is 0 or more instances of
// x. There must be exactly one element in the list.
//
// - Mapping nodes like {a: x, b: y} are defs ([Def]). Any fields not listed are
// implicitly top.
//
// - !sum [x, y, z] is a sum of its children. This can be thought of as a union
// of the values x, y, and z, or as a non-deterministic choice between x, y, and
// z. If a variable appears both inside the sum and outside of it, only the
// non-deterministic choice view really works. The unifier does not directly
// implement sums; instead, this is decoded as a fresh variable that's
// simultaneously bound to x, y, and z.
//
// - !import glob is like a !sum, but its children are read from all files
// matching the given glob pattern, which is interpreted relative to the current
// file path. Each file gets its own variable scope.
func Read(r io.Reader, path string, opts ReadOpts) (Closure, error) {
	dec := yamlDecoder{opts: opts, path: path, env: topEnv}
	v, err := dec.read(r)
	if err != nil {
		return Closure{}, err
	}
	return dec.close(v), nil
}

// ReadFile reads a [Closure] in YAML format from a file.
//
// The file must consist of a single YAML document.
//
// If opts.FS is not set, this sets it to a FS rooted at path's directory.
//
// See [Read] for details.
func ReadFile(path string, opts ReadOpts) (Closure, error) {
	f, err := os.Open(path)
	if err != nil {
		return Closure{}, err
	}
	defer f.Close()

	if opts.FS == nil {
		opts.FS = os.DirFS(filepath.Dir(path))
	}

	return Read(f, path, opts)
}

// UnmarshalYAML implements [yaml.Unmarshaler].
//
// Since there is no way to pass [ReadOpts] to this function, it assumes default
// options.
func (c *Closure) UnmarshalYAML(node *yaml.Node) error {
	dec := yamlDecoder{path: "<yaml.Node>", env: topEnv}
	v, err := dec.root(node)
	if err != nil {
		return err
	}
	*c = dec.close(v)
	return nil
}

type yamlDecoder struct {
	opts ReadOpts
	path string

	vars  map[string]*ident
	nSums int

	env envSet
}

func (dec *yamlDecoder) read(r io.Reader) (*Value, error) {
	n, err := readOneNode(r)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", dec.path, err)
	}

	// Decode YAML node to a Value
	v, err := dec.root(n)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", dec.path, err)
	}

	return v, nil
}

// readOneNode reads a single YAML document from r and returns an error if there
// are more documents in r.
func readOneNode(r io.Reader) (*yaml.Node, error) {
	yd := yaml.NewDecoder(r)

	// Decode as a YAML node
	var node yaml.Node
	if err := yd.Decode(&node); err != nil {
		return nil, err
	}
	np := &node
	if np.Kind == yaml.DocumentNode {
		np = node.Content[0]
	}

	// Ensure there are no more YAML docs in this file
	if err := yd.Decode(nil); err == nil {
		return nil, fmt.Errorf("must not contain multiple documents")
	} else if err != io.EOF {
		return nil, err
	}

	return np, nil
}

// root parses the root of a file.
func (dec *yamlDecoder) root(node *yaml.Node) (*Value, error) {
	// Prepare for variable name resolution in this file. This may be a nested
	// root, so restore the current values when we're done.
	oldVars, oldNSums := dec.vars, dec.nSums
	defer func() {
		dec.vars, dec.nSums = oldVars, oldNSums
	}()
	dec.vars = make(map[string]*ident, 0)
	dec.nSums = 0

	return dec.value(node)
}

// close wraps a decoded [Value] into a [Closure].
func (dec *yamlDecoder) close(v *Value) Closure {
	return Closure{v, dec.env}
}

func (dec *yamlDecoder) value(node *yaml.Node) (vOut *Value, errOut error) {
	pos := &Pos{Path: dec.path, Line: node.Line}

	// Resolve alias nodes.
	if node.Kind == yaml.AliasNode {
		node = node.Alias
	}

	mk := func(d Domain) (*Value, error) {
		v := &Value{Domain: d, pos: pos}
		return v, nil
	}
	mk2 := func(d Domain, err error) (*Value, error) {
		if err != nil {
			return nil, err
		}
		return mk(d)
	}

	// is tests the kind and long tag of node.
	is := func(kind yaml.Kind, tag string) bool {
		return node.Kind == kind && node.LongTag() == tag
	}
	isExact := func() bool {
		if node.Kind != yaml.ScalarNode {
			return false
		}
		// We treat any string-ish YAML node as a string.
		switch node.LongTag() {
		case "!string", "tag:yaml.org,2002:int", "tag:yaml.org,2002:float", "tag:yaml.org,2002:bool", "tag:yaml.org,2002:binary":
			return true
		}
		return false
	}

	// !!str nodes provide a short-hand syntax for several leaf domains that are
	// also available under explicit tags. To simplify checking below, we set
	// strVal to non-"" only for !!str nodes.
	strVal := ""
	isStr := is(yaml.ScalarNode, "tag:yaml.org,2002:str")
	if isStr {
		strVal = node.Value
	}

	switch {
	case is(yaml.ScalarNode, "!var"):
		strVal = "$" + node.Value
		fallthrough
	case strings.HasPrefix(strVal, "$"):
		id, ok := dec.vars[strVal]
		if !ok {
			// We encode different idents with the same string name by adding a
			// #N suffix. Strip that off so it doesn't accumulate. This isn't
			// meant to be used in user-written input, though nothing stops that.
			name, _, _ := strings.Cut(strVal, "#")
			id = &ident{name: name}
			dec.vars[strVal] = id
			dec.env = dec.env.bind(id, topValue)
		}
		return mk(Var{id: id})

	case strVal == "_" || is(yaml.ScalarNode, "!top"):
		return mk(Top{})

	case strVal == "_|_" || is(yaml.ScalarNode, "!bottom"):
		return nil, errors.New("found bottom")

	case isExact():
		val := node.Value
		return mk(NewStringExact(val))

	case isStr || is(yaml.ScalarNode, "!regex"):
		// Any other string we treat as a regex. This will produce an exact
		// string anyway if the regex is literal.
		val := node.Value
		return mk2(NewStringRegex(val))

	case is(yaml.SequenceNode, "!regex"):
		var vals []string
		if err := node.Decode(&vals); err != nil {
			return nil, err
		}
		return mk2(NewStringRegex(vals...))

	case is(yaml.MappingNode, "tag:yaml.org,2002:map"):
		var db DefBuilder
		for i := 0; i < len(node.Content); i += 2 {
			key := node.Content[i]
			if key.Kind != yaml.ScalarNode {
				return nil, fmt.Errorf("non-scalar key %q", key.Value)
			}
			val, err := dec.value(node.Content[i+1])
			if err != nil {
				return nil, err
			}
			db.Add(key.Value, val)
		}
		return mk(db.Build())

	case is(yaml.SequenceNode, "tag:yaml.org,2002:seq"):
		elts := node.Content
		vs := make([]*Value, 0, len(elts))
		for _, elt := range elts {
			v, err := dec.value(elt)
			if err != nil {
				return nil, err
			}
			vs = append(vs, v)
		}
		return mk(NewTuple(vs...))

	case is(yaml.SequenceNode, "!repeat") || is(yaml.SequenceNode, "!repeat-unify"):
		// !repeat must have one child. !repeat-unify is used internally for
		// delayed unification, and is the same, it's just allowed to have more
		// than one child.
		if node.LongTag() == "!repeat" && len(node.Content) != 1 {
			return nil, fmt.Errorf("!repeat must have exactly one child")
		}

		// Decode the children to make sure they're well-formed, but otherwise
		// discard that decoding and do it again every time we need a new
		// element.
		var gen []func(e envSet) (*Value, envSet)
		origEnv := dec.env
		elts := node.Content
		for i, elt := range elts {
			_, err := dec.value(elt)
			if err != nil {
				return nil, err
			}
			// Undo any effects on the environment. We *do* keep any named
			// variables that were added to the vars map in case they were
			// introduced within the element.
			dec.env = origEnv
			// Add a generator function
			gen = append(gen, func(e envSet) (*Value, envSet) {
				dec.env = e
				// TODO: If this is in a sum, this tends to generate a ton of
				// fresh variables that are different on each branch of the
				// parent sum. Does it make sense to hold on to the i'th value
				// of the tuple after we've generated it?
				v, err := dec.value(elts[i])
				if err != nil {
					// It worked the first time, so this really shouldn't hapen.
					panic("decoding repeat element failed")
				}
				return v, dec.env
			})
		}
		return mk(NewRepeat(gen...))

	case is(yaml.SequenceNode, "!sum"):
		vs := make([]*Value, 0, len(node.Content))
		for _, elt := range node.Content {
			v, err := dec.value(elt)
			if err != nil {
				return nil, err
			}
			vs = append(vs, v)
		}
		if len(vs) == 1 {
			return vs[0], nil
		}

		// A sum is implemented as a fresh variable that's simultaneously bound
		// to each of the descendants.
		id := &ident{name: fmt.Sprintf("sum%d", dec.nSums)}
		dec.nSums++
		dec.env = dec.env.bind(id, vs...)
		return mk(Var{id: id})

	case is(yaml.ScalarNode, "!import"):
		if dec.opts.FS == nil {
			return nil, fmt.Errorf("!import not allowed (ReadOpts.FS not set)")
		}
		pat := node.Value

		if !fs.ValidPath(pat) {
			// This will result in Glob returning no results. Give a more useful
			// error message for this case.
			return nil, fmt.Errorf("!import path must not contain '.' or '..'")
		}

		ms, err := fs.Glob(dec.opts.FS, pat)
		if err != nil {
			return nil, fmt.Errorf("resolving !import: %w", err)
		}
		if len(ms) == 0 {
			return nil, fmt.Errorf("!import did not match any files")
		}

		// Parse each file
		vs := make([]*Value, 0, len(ms))
		for _, m := range ms {
			v, err := dec.import1(m)
			if err != nil {
				return nil, err
			}
			vs = append(vs, v)
		}

		// Create a sum.
		if len(vs) == 1 {
			return vs[0], nil
		}
		id := &ident{name: "import"}
		dec.env = dec.env.bind(id, vs...)
		return mk(Var{id: id})
	}

	return nil, fmt.Errorf("unknown node kind %d %v", node.Kind, node.Tag)
}

func (dec *yamlDecoder) import1(path string) (*Value, error) {
	// Make sure we can open the path first.
	f, err := dec.opts.FS.Open(path)
	if err != nil {
		return nil, fmt.Errorf("!import failed: %w", err)
	}
	defer f.Close()

	// Prepare the enter path.
	oldFS, oldPath := dec.opts.FS, dec.path
	defer func() {
		dec.opts.FS, dec.path = oldFS, oldPath
	}()

	// Enter path, which is relative to the current path's directory.
	newPath := filepath.Join(filepath.Dir(dec.path), path)
	subFS, err := fs.Sub(dec.opts.FS, filepath.Dir(path))
	if err != nil {
		return nil, err
	}
	dec.opts.FS, dec.path = subFS, newPath

	// Parse the file.
	return dec.read(f)
}

type yamlEncoder struct {
	idp identPrinter
	e   envSet // We track the environment for !repeat nodes.
}

// TODO: Switch some Value marshaling to Closure?

func (c Closure) MarshalYAML() (any, error) {
	// TODO: If the environment is trivial, just marshal the value.
	enc := &yamlEncoder{}
	return enc.closure(c), nil
}

func (c Closure) String() string {
	b, err := yaml.Marshal(c)
	if err != nil {
		return fmt.Sprintf("marshal failed: %s", err)
	}
	return string(b)
}

func (v *Value) MarshalYAML() (any, error) {
	enc := &yamlEncoder{}
	return enc.value(v), nil
}

func (v *Value) String() string {
	b, err := yaml.Marshal(v)
	if err != nil {
		return fmt.Sprintf("marshal failed: %s", err)
	}
	return string(b)
}

func (enc *yamlEncoder) closure(c Closure) *yaml.Node {
	enc.e = c.env
	var n yaml.Node
	n.Kind = yaml.MappingNode
	n.Tag = "!closure"
	n.Content = make([]*yaml.Node, 4)
	n.Content[0] = new(yaml.Node)
	n.Content[0].SetString("env")
	n.Content[2] = new(yaml.Node)
	n.Content[2].SetString("in")
	n.Content[3] = enc.value(c.val)
	// Fill in the env after we've written the value in case value encoding
	// affects the env.
	n.Content[1] = enc.env(enc.e)
	enc.e = envSet{} // Allow GC'ing the env
	return &n
}

func (enc *yamlEncoder) env(e envSet) *yaml.Node {
	var encode func(e *envExpr) *yaml.Node
	encode = func(e *envExpr) *yaml.Node {
		var n yaml.Node
		switch e.kind {
		default:
			panic("bad kind")
		case envZero:
			n.SetString("0")
		case envUnit:
			n.SetString("1")
		case envBinding:
			var id yaml.Node
			id.SetString(enc.idp.unique(e.id))
			n.Kind = yaml.MappingNode
			n.Content = []*yaml.Node{&id, enc.value(e.val)}
		case envProduct, envSum:
			n.Kind = yaml.SequenceNode
			if e.kind == envProduct {
				n.Tag = "!product"
			} else {
				n.Tag = "!sum"
			}
			for _, e2 := range e.operands {
				n.Content = append(n.Content, encode(e2))
			}
		}
		return &n
	}
	return encode(e.root)
}

var yamlIntRe = regexp.MustCompile(`^-?[0-9]+$`)

func (enc *yamlEncoder) value(v *Value) *yaml.Node {
	var n yaml.Node
	switch d := v.Domain.(type) {
	case nil:
		// Not allowed by unmarshaler, but useful for understanding when
		// something goes horribly wrong.
		//
		// TODO: We might be able to track useful provenance for this, which
		// would really help with debugging unexpected bottoms.
		n.SetString("_|_")
		return &n

	case Top:
		n.SetString("_")
		return &n

	case Def:
		n.Kind = yaml.MappingNode
		for k, elt := range d.All() {
			var kn yaml.Node
			kn.SetString(k)
			n.Content = append(n.Content, &kn, enc.value(elt))
		}
		n.HeadComment = v.PosString()
		return &n

	case Tuple:
		n.Kind = yaml.SequenceNode
		if d.repeat == nil {
			for _, elt := range d.vs {
				n.Content = append(n.Content, enc.value(elt))
			}
		} else {
			if len(d.repeat) == 1 {
				n.Tag = "!repeat"
			} else {
				n.Tag = "!repeat-unify"
			}
			// TODO: I'm not positive this will round-trip everything correctly.
			for _, gen := range d.repeat {
				v, e := gen(enc.e)
				enc.e = e
				n.Content = append(n.Content, enc.value(v))
			}
		}
		return &n

	case String:
		switch d.kind {
		case stringExact:
			n.SetString(d.exact)
			switch {
			// Make this into a "nice" !!int node if I can.
			case yamlIntRe.MatchString(d.exact):
				n.Tag = "tag:yaml.org,2002:int"

			// Or a "nice" !!bool node.
			case d.exact == "false" || d.exact == "true":
				n.Tag = "tag:yaml.org,2002:bool"

			// If this doesn't require escaping, leave it as a str node to avoid
			// the annoying YAML tags. Otherwise, mark it as an exact string.
			// Alternatively, we could always emit a str node with regexp
			// quoting.
			case d.exact != regexp.QuoteMeta(d.exact):
				n.Tag = "!string"
			}
			return &n
		case stringRegex:
			o := make([]string, 0, 1)
			for _, re := range d.re {
				s := re.String()
				s = strings.TrimSuffix(strings.TrimPrefix(s, `\A(?:`), `)\z`)
				o = append(o, s)
			}
			if len(o) == 1 {
				n.SetString(o[0])
				return &n
			}
			n.Encode(o)
			n.Tag = "!regex"
			return &n
		}
		panic("bad String kind")

	case Var:
		// TODO: If Var only appears once in the whole Value and is independent
		// in the environment (part of a term that is only over Var), then emit
		// this as a !sum instead.
		if false {
			var vs []*Value // TODO: Get values of this var.
			if len(vs) == 1 {
				return enc.value(vs[0])
			}
			n.Kind = yaml.SequenceNode
			n.Tag = "!sum"
			for _, elt := range vs {
				n.Content = append(n.Content, enc.value(elt))
			}
			return &n
		}
		n.SetString(enc.idp.unique(d.id))
		if !strings.HasPrefix(d.id.name, "$") {
			n.Tag = "!var"
		}
		return &n
	}
	panic(fmt.Sprintf("unknown domain type %T", v.Domain))
}
