// Package goxview provides GoX's headless persistent component runtime.
package goxview

import (
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
)

type Attr struct {
	Name  string
	Value any
}
type Node struct {
	Tag, Text string
	Attrs     []Attr
	Children  []Node
	Component *Definition
	Key       any
	provider  *provider
}
type Definition struct {
	ID     string
	Render func(*Instance, []Attr, []Node) Node
}

func Define(id string, f func(*Instance, []Attr, []Node) Node) *Definition { return &Definition{id, f} }
func Element(tag string, attrs []Attr, children []Node) Node {
	n := Node{Tag: tag, Children: children}
	for _, a := range attrs {
		if a.Name == "key" {
			n.Key = a.Value
		} else {
			n.Attrs = append(n.Attrs, a)
		}
	}
	return n
}
func Component(d *Definition, a []Attr, c []Node) Node {
	if d == nil {
		panic("nil component")
	}
	n := Element("", a, c)
	n.Component = d
	return n
}
func Text(s string) Node { return Node{Tag: "#text", Text: s} }
func Child(v any) []Node {
	switch v := v.(type) {
	case nil:
		return nil
	case Node:
		return []Node{v}
	case []Node:
		return v
	case string:
		return []Node{Text(v)}
	default:
		panic(fmt.Sprintf("unsupported child %T", v))
	}
}
func Prop[T any](a []Attr, name string) T {
	for _, p := range a {
		if p.Name == name {
			if v, ok := p.Value.(T); ok {
				return v
			}
			if p.Value == nil {
				var z T
				switch reflect.TypeOf((*T)(nil)).Elem().Kind() {
				case reflect.Pointer, reflect.Slice, reflect.Map, reflect.Func, reflect.Interface, reflect.Chan:
					return z
				}
			}
			panic("wrong property type: " + name)
		}
	}
	panic("missing property: " + name)
}

type Instance struct {
	root     *Root
	alive    atomic.Bool
	slots    map[string]any
	contexts map[uint64]any
}
type update struct {
	i *Instance
	f func()
}
type Root struct {
	mu           sync.Mutex
	queue        []update
	rendering    atomic.Bool
	instances    map[string]*Instance
	source, tree Node
	commit       func(Node)
	wake         func()
}

func NewRoot(commit func(Node), wake func()) *Root {
	return &Root{instances: map[string]*Instance{}, commit: commit, wake: wake}
}
func (r *Root) Tree() Node { return r.tree }

// Mount, Flush, Tree and Unmount belong to one host goroutine. State setters
// offer explicit Enqueue methods for workers that can overlap rendering.
func (r *Root) Mount(n Node) error {
	old := r.source
	r.source = n
	if err := r.render(); err != nil {
		r.source = old
		return err
	}
	return nil
}
func (r *Root) Unmount() {
	for _, i := range r.instances {
		i.alive.Store(false)
	}
	r.instances = map[string]*Instance{}
	r.mu.Lock()
	r.queue = nil
	r.mu.Unlock()
	r.source = Node{}
	r.tree = Node{}
	if r.commit != nil {
		r.commit(r.tree)
	}
}
func (r *Root) enqueue(i *Instance, async bool, f func()) {
	if !i.alive.Load() {
		return
	}
	if !async && r.rendering.Load() {
		panic("state update during render; workers must use Enqueue or EnqueueUpdate")
	}
	r.mu.Lock()
	r.queue = append(r.queue, update{i, f})
	r.mu.Unlock()
	if r.wake != nil {
		r.wake()
	}
}

// Flush processes one batch; work queued by commit is handled by the next call.
func (r *Root) Flush() (err error) {
	r.mu.Lock()
	q := r.queue
	r.queue = nil
	r.mu.Unlock()
	dirty := false
	nextUpdate := 0
	defer func() {
		if v := recover(); v != nil {
			r.mu.Lock()
			r.queue = append(q[nextUpdate:], r.queue...)
			r.mu.Unlock()
			err = fmt.Errorf("goxview: state update: %v", v)
		}
	}()
	for index, u := range q {
		nextUpdate = index + 1
		if u.i.alive.Load() {
			u.f()
			dirty = true
		}
	}
	if dirty {
		return r.render()
	}
	return nil
}
func key(v any, index int) string {
	if v == nil {
		return fmt.Sprint("p", index)
	}
	if s, ok := v.(string); ok {
		return fmt.Sprintf("s%d:%s", len(s), s)
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return fmt.Sprint("i", rv.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return fmt.Sprint("u", rv.Uint())
	}
	panic("key must be string or integer")
}
func (r *Root) render() (err error) {
	if !r.rendering.CompareAndSwap(false, true) {
		return fmt.Errorf("recursive render")
	}
	defer r.rendering.Store(false)
	next := map[string]*Instance{}
	defer func() {
		if v := recover(); v != nil {
			for p, i := range next {
				if r.instances[p] != i {
					i.alive.Store(false)
				}
			}
			err = fmt.Errorf("goxview: %v", v)
		}
	}()
	tree := r.resolve(r.source, "root/"+key(r.source.Key, 0), map[uint64]any{}, next)
	for p, i := range r.instances {
		if next[p] != i {
			i.alive.Store(false)
		}
	}
	r.instances = next
	r.tree = tree
	r.rendering.Store(false)
	if r.commit != nil {
		r.commit(tree)
	}
	return nil
}
func (r *Root) resolve(n Node, path string, ctx map[uint64]any, next map[string]*Instance) Node {
	if n.provider != nil {
		c := map[uint64]any{}
		for k, v := range ctx {
			c[k] = v
		}
		c[n.provider.id] = n.provider.value
		ctx = c
	}
	if n.Component != nil {
		p := fmt.Sprintf("%s/c%d:%s", path, len(n.Component.ID), n.Component.ID)
		i := r.instances[p]
		if i == nil {
			i = &Instance{root: r, slots: map[string]any{}}
			i.alive.Store(true)
		}
		next[p] = i
		i.contexts = ctx
		return r.resolve(n.Component.Render(i, n.Attrs, n.Children), p+"/body", ctx, next)
	}
	path += fmt.Sprintf("/element%d:%s", len(n.Tag), n.Tag)
	out := n
	out.Children = nil
	seen := map[string]bool{}
	for j, c := range n.Children {
		k := key(c.Key, j)
		if seen[k] {
			panic("duplicate sibling key")
		}
		seen[k] = true
		out.Children = append(out.Children, r.resolve(c, fmt.Sprintf("%s/%d:%s", path, len(k), k), ctx, next))
	}
	return out
}
func requireRender(i *Instance) {
	if i == nil || !i.root.rendering.Load() {
		panic("hook outside render")
	}
}

type state[T any] struct{ value T }

// StateSetter is callable with a replacement and supports Update on latest state.
// A private protocol closure transports either replacement or updater requests.
type stateRequest[T any] struct {
	transform func(T) T
	async     bool
}
type StateSetter[T any] func(T, ...stateRequest[T])

func (s StateSetter[T]) Update(f func(T) T) { var z T; s(z, stateRequest[T]{transform: f}) }

// Enqueue schedules a replacement from a worker, including during a render.
func (s StateSetter[T]) Enqueue(v T) { s(v, stateRequest[T]{async: true}) }

// EnqueueUpdate schedules a functional update from a worker.
func (s StateSetter[T]) EnqueueUpdate(f func(T) T) {
	var z T
	s(z, stateRequest[T]{transform: f, async: true})
}
func UseState[T any](i *Instance, slot string, initial T) (T, StateSetter[T]) {
	requireRender(i)
	var s *state[T]
	if old, ok := i.slots[slot]; ok {
		var valid bool
		s, valid = old.(*state[T])
		if !valid {
			panic("hook type changed")
		}
	} else {
		s = &state[T]{initial}
		i.slots[slot] = s
	}
	set := StateSetter[T](func(v T, requests ...stateRequest[T]) {
		var request stateRequest[T]
		if len(requests) > 0 {
			request = requests[0]
		}
		i.root.enqueue(i, request.async, func() {
			if request.transform != nil {
				s.value = request.transform(s.value)
			} else {
				s.value = v
			}
		})
	})
	return s.value, set
}

type Ref[T any] struct{ Current T }

func UseRef[T any](i *Instance, slot string, initial T) *Ref[T] {
	requireRender(i)
	if old, ok := i.slots[slot]; ok {
		if v, ok := old.(*Ref[T]); ok {
			return v
		}
		panic("hook type changed")
	}
	v := &Ref[T]{initial}
	i.slots[slot] = v
	return v
}

var contextIDs atomic.Uint64

type Context[T any] struct {
	id       uint64
	fallback T
}
type provider struct {
	id    uint64
	value any
}

func NewContext[T any](fallback T) *Context[T] { return &Context[T]{contextIDs.Add(1), fallback} }
func (c *Context[T]) Provider(value T, children ...Node) Node {
	return Node{Tag: "#provider", Children: children, provider: &provider{c.id, value}}
}
func UseContext[T any](i *Instance, c *Context[T]) T {
	requireRender(i)
	if v, ok := i.contexts[c.id]; ok {
		if v == nil {
			var z T
			return z
		}
		return v.(T)
	}
	return c.fallback
}
