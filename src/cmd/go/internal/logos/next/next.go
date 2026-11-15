// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package next

import (
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"internal/logosfile"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"
)

var CmdNext = &base.Command{
	Run:       runNext,
	UsageLine: "logos next [--fix | --ai-patch]",
	Short:     "convert a Go module to a loGo/s module",
	Long: `
Next converts the whole Go module in the current directory to a loGo/s
module: plain next prints a unified diff and changes nothing, next
--fix renames every Go file to .lgo and rewrites matched patterns,
and next --ai-patch writes and prints next_tasks.json without
renaming anything.

Matched patterns become ? checks, $"..." interpolations, and map
chains through byte-span splicing: every comment and build line
outside rewrites stays byte-identical. Imports of Go packages outside
the module stay as written. Complex refactors stay unchanged and are
listed in next_tasks.json for AI patching.
	`,
}

var nextFix bool

var nextAIPatch bool

func init() {
	CmdNext.Flag.BoolVar(&nextFix, "fix", false, "")
	CmdNext.Flag.BoolVar(&nextAIPatch, "ai-patch", false, "")
	base.AddChdirFlag(&CmdNext.Flag)
}

func Register(root *base.Command) {
	root.Commands = append(root.Commands, CmdNext)
}

type wireEdit struct {
	StartLine   int    `json:"start_line"`
	StartCol    int    `json:"start_col"`
	EndLine     int    `json:"end_line"`
	EndCol      int    `json:"end_col"`
	Replacement string `json:"replacement"`
}

type wireTask struct {
	Category    string `json:"category"`
	Rule        string `json:"rule"`
	StartLine   int    `json:"start_line"`
	EndLine     int    `json:"end_line"`
	FuncSig     string `json:"function_signature"`
	Instruction string `json:"ai_instruction"`
}

type wireFile struct {
	File  string     `json:"file"`
	Sum   string     `json:"sha256"`
	Edits []wireEdit `json:"edits"`
	Tasks []wireTask `json:"tasks"`
}

type wireReport struct {
	Version string     `json:"version"`
	Package string     `json:"package"`
	Files   []wireFile `json:"files"`
}

type taskSpan struct {
	Start int `json:"start_line"`
	End   int `json:"end_line"`
}

type planTask struct {
	ID        string   `json:"id"`
	Category  string   `json:"category"`
	File      string   `json:"file"`
	Span      taskSpan `json:"span"`
	Rule      string   `json:"rule"`
	FuncSig   string   `json:"function_signature"`
	AiInstr   string   `json:"ai_instruction"`
	VerifyCmd string   `json:"verification_cmd"`
}

type planFile struct {
	Version string     `json:"version"`
	Tasks   []planTask `json:"tasks"`
}

type fileWork struct {
	rel   string
	data  []byte
	sum   string
	edits []wireEdit
	tasks []wireTask
}

func runNext(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) > 0 {
		base.Fatalf("logos next: unexpected package arguments")
	}
	if nextFix && nextAIPatch {
		base.Fatalf("logos next: --fix and --ai-patch are exclusive")
	}
	gomod := modload.FindGoMod(base.Cwd())
	if gomod == "" {
		base.Fatalf("logos next: no main module (run inside a module)")
	}
	moddir := filepath.Dir(gomod)
	work := scanModule(ctx, moddir)
	switch {
	case nextFix:
		work.writePlan(moddir)
		work.apply(moddir)
	case nextAIPatch:
		work.writePlan(moddir)
		fmt.Printf("%s", work.planData())
	default:
		work.printDiff()
	}
}

type scanState struct {
	files map[string]*fileWork
	order []string
	tasks []planTask
}

func scanModule(ctx context.Context, moddir string) *scanState {
	staging, err := os.MkdirTemp("", "logosnext")
	if err != nil {
		base.Fatalf("logos next: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(staging); err != nil {
			base.Errorf("logos next: %v", err)
		}
	}()
	sidecar := filepath.Join(staging, "next.jsonl")
	moduleLoader := modload.NewLoader()
	moduleLoader.InitWorkfile()
	work.BuildInit(moduleLoader)
	load.ArmLogosNext(sidecar)
	defer load.DisarmLogosNext()
	opts := load.PackageOpts{AutoVCS: true}
	pkgs := load.PackagesAndErrors(moduleLoader, ctx, opts, []string{"all"})
	load.CheckPackageErrors(pkgs)
	roots := nextRoots(pkgs)
	if len(roots) == 0 {
		if hasLgoFiles(moddir) {
			fmt.Println("logos next: already converted")
			base.Exit()
		}
		base.Fatalf("logos next: no Go packages to convert")
	}
	buildNextRoots(moduleLoader, ctx, roots)
	buildNextTests(moduleLoader, ctx, opts, roots)
	state := &scanState{files: map[string]*fileWork{}}
	state.readSidecar(sidecar, moddir)
	return state
}

func nextRoots(pkgs []*load.Package) []*load.Package {
	var roots []*load.Package
	for _, p := range load.PackageList(pkgs) {
		if !nextRootKeep(p) {
			continue
		}
		roots = append(roots, p)
	}
	slices.SortFunc(roots, func(a, b *load.Package) int { return cmp.Compare(a.ImportPath, b.ImportPath) })
	return roots
}

func nextRootKeep(p *load.Package) bool {
	if p.Module == nil {
		return false
	}
	if !p.Module.Main {
		return false
	}
	if p.Standard {
		return false
	}
	if p.IsLogos() {
		return false
	}
	if p.ForTest != "" {
		return false
	}
	if p.Internal.TestmainGo != nil {
		return false
	}
	return load.HasNextFiles(p)
}

func doBuild(b *work.Builder, moduleLoader *modload.Loader, ctx context.Context, roots []*load.Package) {
	root := &work.Action{Mode: "go build"}
	for _, p := range roots {
		root.Deps = append(root.Deps, b.AutoAction(moduleLoader, work.ModeBuild, work.ModeBuild, p))
	}
	b.Do(ctx, root)
	if root.Failed != nil {
		base.SetExitStatus(1)
	}
	if base.GetExitStatus() != 0 {
		base.Exit()
	}
}

func buildNextRoots(moduleLoader *modload.Loader, ctx context.Context, roots []*load.Package) {
	b := work.NewBuilder("", moduleLoader.VendorDirOrEmpty)
	defer func() {
		if err := b.Close(); err != nil {
			base.Fatalf("logos next: %v", err)
		}
	}()
	doBuild(b, moduleLoader, ctx, roots)
}

func buildNextTests(moduleLoader *modload.Loader, ctx context.Context, opts load.PackageOpts, roots []*load.Package) {
	b := work.NewBuilder("", moduleLoader.VendorDirOrEmpty)
	defer func() {
		if err := b.Close(); err != nil {
			base.Fatalf("logos next: %v", err)
		}
	}()
	var mains []*load.Package
	for _, p := range roots {
		pmain := buildNextTestMain(ctx, moduleLoader, opts, p)
		if pmain == nil {
			continue
		}
		if err := b.StageTestmain(pmain); err != nil {
			base.Fatalf("logos next: %v", err)
		}
		mains = append(mains, pmain)
	}
	if len(mains) == 0 {
		return
	}
	doBuild(b, moduleLoader, ctx, mains)
}

func buildNextTestMain(ctx context.Context, moduleLoader *modload.Loader, opts load.PackageOpts, p *load.Package) *load.Package {
	if len(p.TestGoFiles) == 0 && len(p.XTestGoFiles) == 0 {
		return nil
	}
	pmain, _, _, perr := load.TestPackagesFor(moduleLoader, ctx, opts, p, nil)
	if perr != nil {
		base.Fatalf("logos next: %v", perr)
	}
	if pmain == nil {
		base.Fatalf("logos next: no test main for %s", p.ImportPath)
	}
	return pmain
}

func (s *scanState) readSidecar(sidecar, moddir string) {
	known := walkGoFiles(moddir)
	lines := readLines(sidecar)
	byFile := map[string]*fileWork{}
	for _, line := range lines {
		var rep wireReport
		if err := json.Unmarshal([]byte(line), &rep); err != nil {
			base.Fatalf("logos next: bad sidecar: %v", err)
		}
		for _, f := range rep.Files {
			rel, ok := known[filepath.Clean(f.File)]
			if !ok {
				continue
			}
			w := byFile[rel]
			if w == nil {
				data, err := os.ReadFile(filepath.Join(moddir, rel))
				if err != nil {
					base.Fatalf("logos next: %v", err)
				}
				sum := sha256.Sum256(data)
				if hex.EncodeToString(sum[:]) != f.Sum {
					base.Fatalf("logos next: %s changed during scan", rel)
				}
				w = &fileWork{rel: rel, data: data, sum: f.Sum}
				byFile[rel] = w
			}
			w.edits = append(w.edits, f.Edits...)
			w.tasks = append(w.tasks, f.Tasks...)
		}
	}
	for rel, w := range byFile {
		w.edits = dedupEdits(w.edits)
		checkOverlaps(rel, w.edits)
		s.files[rel] = w
		s.order = append(s.order, rel)
	}
	slices.Sort(s.order)
	s.assignTasks()
}

func walkGoFiles(moddir string) map[string]string {
	known := map[string]string{}
	err := filepath.WalkDir(moddir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if walkNextSkipDir(path, moddir) {
				return filepath.SkipDir
			}
			return nil
		}
		if !walkNextFileOK(path) {
			return nil
		}
		return walkNextCollect(known, moddir, path)
	})
	if err != nil {
		base.Fatalf("logos next: %v", err)
	}
	return known
}

func walkNextSkipDir(path, moddir string) bool {
	if path == moddir {
		return false
	}
	dirname := filepath.Base(path)
	if dirname == "vendor" || dirname == "testdata" || strings.HasPrefix(dirname, ".") {
		return true
	}
	if st, err := os.Stat(filepath.Join(path, "go.mod")); err == nil && !st.IsDir() {
		return true
	}
	return false
}
func hasLgoFiles(moddir string) bool {
	found := false
	if err := filepath.WalkDir(moddir, func(path string, d os.DirEntry, err error) error {
		if err != nil || found {
			return err
		}
		if d.IsDir() {
			if walkNextSkipDir(path, moddir) {
				return filepath.SkipDir
			}
			return nil
		}
		if logosfile.Is(path) {
			found = true
		}
		return nil
	}); err != nil {
		base.Fatalf("logos next: %v", err)
	}
	return found
}

func walkNextFileOK(path string) bool {
	if logosfile.Is(path) {
		base.Fatalf("logos next: module already holds loGo/s file %s", path)
	}
	if !strings.HasSuffix(path, ".go") {
		return false
	}
	name := filepath.Base(path)
	if strings.HasPrefix(name, "_") || strings.HasPrefix(name, ".") {
		return false
	}
	return true
}

func walkNextCollect(known map[string]string, moddir, path string) error {
	abs, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	real, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return err
	}
	modReal, err := filepath.EvalSymlinks(moddir)
	if err != nil {
		return err
	}
	relReal, err := filepath.Rel(modReal, real)
	if err != nil {
		return err
	}
	if relReal == ".." || strings.HasPrefix(relReal, ".."+string(filepath.Separator)) {
		return nil
	}
	rel, err := filepath.Rel(moddir, abs)
	if err != nil {
		return err
	}
	known[filepath.Clean(abs)] = rel
	return nil
}

func readLines(sidecar string) []string {
	data, err := os.ReadFile(sidecar)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		base.Fatalf("logos next: %v", err)
	}
	var lines []string
	for line := range strings.SplitSeq(string(data), "\n") {
		if strings.TrimSpace(line) != "" {
			lines = append(lines, line)
		}
	}
	return lines
}

func dedupEdits(edits []wireEdit) []wireEdit {
	seen := map[wireEdit]bool{}
	var out []wireEdit
	for _, e := range edits {
		if seen[e] {
			continue
		}
		seen[e] = true
		out = append(out, e)
	}
	slices.SortFunc(out, func(a, b wireEdit) int {
		return cmp.Or(cmp.Compare(a.StartLine, b.StartLine), cmp.Compare(a.StartCol, b.StartCol))
	})
	return out
}

func checkOverlaps(rel string, edits []wireEdit) {
	for i := 1; i < len(edits); i++ {
		prev := edits[i-1]
		cur := edits[i]
		if cur.StartLine < prev.EndLine || (cur.StartLine == prev.EndLine && cur.StartCol < prev.EndCol) {
			base.Fatalf("logos next: overlapping rewrites in %s", rel)
		}
	}
}

func (s *scanState) assignTasks() {
	seen := map[planTask]bool{}
	n := 0
	for _, rel := range s.order {
		w := s.files[rel]
		slices.SortFunc(w.tasks, func(a, b wireTask) int {
			return cmp.Or(cmp.Compare(a.StartLine, b.StartLine), cmp.Compare(a.EndLine, b.EndLine))
		})
		for _, t := range w.tasks {
			task := planTask{
				Category:  t.Category,
				File:      lgoName(rel),
				Span:      taskSpan{Start: remap(w.edits, t.StartLine), End: remap(w.edits, t.EndLine)},
				Rule:      t.Rule,
				FuncSig:   t.FuncSig,
				AiInstr:   t.Instruction,
				VerifyCmd: "logos build ./...",
			}
			if seen[task] {
				continue
			}
			seen[task] = true
			n++
			task.ID = fmt.Sprintf("next-%03d", n)
			s.tasks = append(s.tasks, task)
		}
	}
}

func remap(edits []wireEdit, line int) int {
	orig := line
	delta := 0
	for _, e := range edits {
		if e.EndLine < orig {
			delta += newLines(e.Replacement) - (e.EndLine - e.StartLine + 1)
		}
	}
	return orig + delta
}

func newLines(s string) int {
	return strings.Count(s, "\n") + 1
}

func lgoName(rel string) string {
	return strings.TrimSuffix(rel, ".go") + logosfile.Canonical
}

func (s *scanState) planData() []byte {
	plan := planFile{Version: "logos-next-1", Tasks: s.tasks}
	if plan.Tasks == nil {
		plan.Tasks = []planTask{}
	}
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(plan); err != nil {
		base.Fatalf("logos next: %v", err)
	}
	return buf.Bytes()
}

func (s *scanState) writePlan(moddir string) {
	if err := os.WriteFile(filepath.Join(moddir, "next_tasks.json"), s.planData(), 0666); err != nil {
		base.Fatalf("logos next: %v", err)
	}
}

func applyEdits(data []byte, edits []wireEdit) []byte {
	starts := lineStartsOf(data)
	out := data
	for i := len(edits) - 1; i >= 0; i-- {
		e := edits[i]
		start := offsetOf(starts, len(data), e.StartLine, e.StartCol)
		end := offsetOf(starts, len(data), e.EndLine, e.EndCol)
		if start < 0 || end < start || end > len(out) {
			base.Fatalf("logos next: rewrite out of range %d:%d-%d:%d", e.StartLine, e.StartCol, e.EndLine, e.EndCol)
		}
		part := make([]byte, 0, len(out)-(end-start)+len(e.Replacement))
		part = append(part, out[:start]...)
		part = append(part, e.Replacement...)
		part = append(part, out[end:]...)
		out = part
	}
	return out
}

func lineStartsOf(data []byte) []int {
	starts := []int{0}
	for i, c := range data {
		if c == '\n' {
			starts = append(starts, i+1)
		}
	}
	return starts
}

func offsetOf(starts []int, size, line, col int) int {
	if line < 1 || line > len(starts) || col < 1 {
		return -1
	}
	end := size
	if line < len(starts) {
		end = starts[line] - 1
		if end > size {
			end = size
		}
	}
	off := starts[line-1] + col - 1
	if off > end {
		return -1
	}
	return off
}

func applySumOk(data []byte, sum string) bool {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:]) == sum
}
func (s *scanState) apply(moddir string) {
	known := walkGoFiles(moddir)
	rels := slices.Sorted(maps.Values(known))
	rewrites := 0
	for _, rel := range rels {
		abs := filepath.Join(moddir, rel)
		data, err := os.ReadFile(abs)
		if err != nil {
			base.Fatalf("logos next: %v", err)
		}
		w := s.files[rel]
		if w != nil && !applySumOk(data, w.sum) {
			base.Fatalf("logos next: %s changed during apply", rel)
		}
		target := filepath.Join(moddir, lgoName(rel))
		if w == nil || len(w.edits) == 0 {
			if err := os.Rename(abs, target); err != nil {
				base.Fatalf("logos next: %v", err)
			}
			continue
		}
		out := applyEdits(data, w.edits)
		if err := os.WriteFile(target, out, 0666); err != nil {
			base.Fatalf("logos next: %v", err)
		}
		if err := os.Remove(abs); err != nil {
			base.Fatalf("logos next: %v", err)
		}
		rewrites += len(w.edits)
	}
	fmt.Printf("logos next: converted %d files with %d rewrites; %d tasks in next_tasks.json\n", len(rels), rewrites, len(s.tasks))
}

func (s *scanState) printDiff() {
	shown := false
	for _, rel := range s.order {
		w := s.files[rel]
		if len(w.edits) == 0 {
			continue
		}
		shown = true
		fmt.Printf("--- a/%s\n+++ b/%s\n", rel, lgoName(rel))
		for _, hunk := range hunks(w) {
			fmt.Print(hunk)
		}
	}
	if !shown {
		fmt.Println("logos next: no conversions found")
	}
}

const diffContext = 3

type lineSpan struct{ s, e int }

func hunks(w *fileWork) []string {
	oldLines := fileLines(w.data)
	newLines := fileLines(applyEdits(w.data, w.edits))
	var olds []lineSpan
	for _, e := range w.edits {
		olds = append(olds, lineSpan{e.StartLine, e.EndLine})
	}
	var out []string
	for _, m := range mergeSpans(olds) {
		ns := remap(w.edits, m.s)
		ne := ns + (m.e - m.s) + deltaLines(w.edits, m.s, m.e)
		os := clamp(m.s-diffContext, len(oldLines))
		oe := clamp(m.e+diffContext, len(oldLines))
		ns2 := clamp(ns-diffContext, len(newLines))
		ne2 := clamp(ne+diffContext, len(newLines))
		var b strings.Builder
		fmt.Fprintf(&b, "@@ -%d,%d +%d,%d @@\n", os, oe-os+1, ns2, ne2-ns2+1)
		for _, op := range diffOps(oldLines[os-1:oe], newLines[ns2-1:ne2]) {
			switch op.kind {
			case ' ':
				fmt.Fprintf(&b, " %s\n", op.a)
			case '-':
				fmt.Fprintf(&b, "-%s\n", op.a)
			case '+':
				fmt.Fprintf(&b, "+%s\n", op.b)
			}
		}
		out = append(out, b.String())
	}
	return out
}

func fileLines(data []byte) []string {
	return strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
}

type diffOp struct {
	kind byte
	a    string
	b    string
}

func diffOps(a, b []string) []diffOp {
	rev := diffBacktrack(a, b, diffTable(a, b))
	out := make([]diffOp, 0, len(rev))
	for k := len(rev) - 1; k >= 0; k-- {
		out = append(out, rev[k])
	}
	return out
}

func diffTable(a, b []string) []int {
	n, m := len(a), len(b)
	width := m + 1
	dp := make([]int, (n+1)*width)
	at := func(i, j int) int { return dp[i*width+j] }
	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			if a[i-1] == b[j-1] {
				dp[i*width+j] = at(i-1, j-1) + 1
				continue
			}
			if at(i-1, j) >= at(i, j-1) {
				dp[i*width+j] = at(i-1, j)
			} else {
				dp[i*width+j] = at(i, j-1)
			}
		}
	}
	return dp
}

func diffBacktrack(a, b []string, dp []int) []diffOp {
	n, m := len(a), len(b)
	width := m + 1
	at := func(i, j int) int { return dp[i*width+j] }
	var rev []diffOp
	i, j := n, m
	for i > 0 && j > 0 {
		if a[i-1] == b[j-1] {
			rev = append(rev, diffOp{kind: ' ', a: a[i-1]})
			i--
			j--
			continue
		}
		if at(i-1, j) > at(i, j-1) {
			rev = append(rev, diffOp{kind: '-', a: a[i-1]})
			i--
			continue
		}
		rev = append(rev, diffOp{kind: '+', b: b[j-1]})
		j--
	}
	for ; i > 0; i-- {
		rev = append(rev, diffOp{kind: '-', a: a[i-1]})
	}
	for ; j > 0; j-- {
		rev = append(rev, diffOp{kind: '+', b: b[j-1]})
	}
	return rev
}

func deltaLines(edits []wireEdit, s, e int) int {
	d := 0
	for _, ed := range edits {
		if ed.StartLine >= s && ed.EndLine <= e {
			d += newLines(ed.Replacement) - (ed.EndLine - ed.StartLine + 1)
		}
	}
	return d
}

func mergeSpans(spans []lineSpan) []lineSpan {
	if len(spans) == 0 {
		return nil
	}
	out := spans[:1]
	for _, sp := range spans[1:] {
		last := &out[len(out)-1]
		if sp.s-last.e-1 <= 2*diffContext {
			if sp.e > last.e {
				last.e = sp.e
			}
			continue
		}
		out = append(out, sp)
	}
	return out
}

func clamp(v, hi int) int {
	if v < 1 {
		return 1
	}
	if v > hi {
		return hi
	}
	return v
}
