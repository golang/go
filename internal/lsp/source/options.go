// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/asmdecl"
	"golang.org/x/tools/go/analysis/passes/assign"
	"golang.org/x/tools/go/analysis/passes/atomic"
	"golang.org/x/tools/go/analysis/passes/atomicalign"
	"golang.org/x/tools/go/analysis/passes/bools"
	"golang.org/x/tools/go/analysis/passes/buildtag"
	"golang.org/x/tools/go/analysis/passes/cgocall"
	"golang.org/x/tools/go/analysis/passes/composite"
	"golang.org/x/tools/go/analysis/passes/copylock"
	"golang.org/x/tools/go/analysis/passes/deepequalerrors"
	"golang.org/x/tools/go/analysis/passes/errorsas"
	"golang.org/x/tools/go/analysis/passes/fieldalignment"
	"golang.org/x/tools/go/analysis/passes/httpresponse"
	"golang.org/x/tools/go/analysis/passes/ifaceassert"
	"golang.org/x/tools/go/analysis/passes/loopclosure"
	"golang.org/x/tools/go/analysis/passes/lostcancel"
	"golang.org/x/tools/go/analysis/passes/nilfunc"
	"golang.org/x/tools/go/analysis/passes/printf"
	"golang.org/x/tools/go/analysis/passes/shift"
	"golang.org/x/tools/go/analysis/passes/sortslice"
	"golang.org/x/tools/go/analysis/passes/stdmethods"
	"golang.org/x/tools/go/analysis/passes/stringintconv"
	"golang.org/x/tools/go/analysis/passes/structtag"
	"golang.org/x/tools/go/analysis/passes/testinggoroutine"
	"golang.org/x/tools/go/analysis/passes/tests"
	"golang.org/x/tools/go/analysis/passes/unmarshal"
	"golang.org/x/tools/go/analysis/passes/unreachable"
	"golang.org/x/tools/go/analysis/passes/unsafeptr"
	"golang.org/x/tools/go/analysis/passes/unusedresult"
	"golang.org/x/tools/internal/lsp/analysis/fillreturns"
	"golang.org/x/tools/internal/lsp/analysis/fillstruct"
	"golang.org/x/tools/internal/lsp/analysis/nonewvars"
	"golang.org/x/tools/internal/lsp/analysis/noresultvalues"
	"golang.org/x/tools/internal/lsp/analysis/simplifycompositelit"
	"golang.org/x/tools/internal/lsp/analysis/simplifyrange"
	"golang.org/x/tools/internal/lsp/analysis/simplifyslice"
	"golang.org/x/tools/internal/lsp/analysis/undeclaredname"
	"golang.org/x/tools/internal/lsp/analysis/unusedparams"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/diff/myers"
	"golang.org/x/tools/internal/lsp/protocol"
	errors "golang.org/x/xerrors"
)

var (
	optionsOnce    sync.Once
	defaultOptions *Options
)

//go:generate go run golang.org/x/tools/internal/lsp/source/genapijson -output api_json.go

// DefaultOptions is the options that are used for Gopls execution independent
// of any externally provided configuration (LSP initialization, command
// invokation, etc.).
func DefaultOptions() *Options {
	optionsOnce.Do(func() {
		var commands []string
		for _, c := range Commands {
			commands = append(commands, c.ID())
		}
		defaultOptions = &Options{
			ClientOptions: ClientOptions{
				InsertTextFormat:                  protocol.PlainTextTextFormat,
				PreferredContentFormat:            protocol.Markdown,
				ConfigurationSupported:            true,
				DynamicConfigurationSupported:     true,
				DynamicWatchedFilesSupported:      true,
				LineFoldingOnly:                   false,
				HierarchicalDocumentSymbolSupport: true,
			},
			ServerOptions: ServerOptions{
				SupportedCodeActions: map[FileKind]map[protocol.CodeActionKind]bool{
					Go: {
						protocol.SourceFixAll:          true,
						protocol.SourceOrganizeImports: true,
						protocol.QuickFix:              true,
						protocol.RefactorRewrite:       true,
						protocol.RefactorExtract:       true,
					},
					Mod: {
						protocol.SourceOrganizeImports: true,
					},
					Sum: {},
				},
				SupportedCommands: commands,
			},
			UserOptions: UserOptions{
				HoverKind:  FullDocumentation,
				LinkTarget: "pkg.go.dev",
				Codelenses: map[string]bool{
					CommandGenerate.Name:          true,
					CommandRegenerateCgo.Name:     true,
					CommandTidy.Name:              true,
					CommandToggleDetails.Name:     false,
					CommandUpgradeDependency.Name: true,
					CommandVendor.Name:            true,
				},
				LinksInHover:   true,
				ImportShortcut: Both,
				Matcher:        Fuzzy,
				SymbolMatcher:  SymbolFuzzy,
				SymbolStyle:    DynamicSymbols,
			},
			DebuggingOptions: DebuggingOptions{
				CompletionBudget: 100 * time.Millisecond,
			},
			ExperimentalOptions: ExperimentalOptions{
				ExpandWorkspaceToModule:     true,
				ExperimentalPackageCacheKey: true,
			},
			InternalOptions: InternalOptions{
				LiteralCompletions:      true,
				TempModfile:             true,
				CompleteUnimported:      true,
				CompletionDocumentation: true,
				DeepCompletion:          true,
			},
			Hooks: Hooks{
				ComputeEdits:         myers.ComputeEdits,
				URLRegexp:            urlRegexp(),
				DefaultAnalyzers:     defaultAnalyzers(),
				TypeErrorAnalyzers:   typeErrorAnalyzers(),
				ConvenienceAnalyzers: convenienceAnalyzers(),
				StaticcheckAnalyzers: map[string]Analyzer{},
				GoDiff:               true,
			},
		}
	})
	return defaultOptions
}

// Options holds various configuration that affects Gopls execution, organized
// by the nature or origin of the settings.
type Options struct {
	ClientOptions
	ServerOptions
	UserOptions
	DebuggingOptions
	ExperimentalOptions
	InternalOptions
	Hooks
}

// ClientOptions holds LSP-specific configuration that is provided by the
// client.
type ClientOptions struct {
	InsertTextFormat                  protocol.InsertTextFormat
	ConfigurationSupported            bool
	DynamicConfigurationSupported     bool
	DynamicWatchedFilesSupported      bool
	PreferredContentFormat            protocol.MarkupKind
	LineFoldingOnly                   bool
	HierarchicalDocumentSymbolSupport bool
	SemanticTypes                     []string
	SemanticMods                      []string
}

// ServerOptions holds LSP-specific configuration that is provided by the
// server.
type ServerOptions struct {
	SupportedCodeActions map[FileKind]map[protocol.CodeActionKind]bool
	SupportedCommands    []string
}

// UserOptions holds custom Gopls configuration (not part of the LSP) that is
// modified by the client.
type UserOptions struct {
	// BuildFlags is the set of flags passed on to the build system when invoked.
	// It is applied to queries like `go list`, which is used when discovering files.
	// The most common use is to set `-tags`.
	BuildFlags []string

	// Env adds environment variables to external commands run by `gopls`, most notably `go list`.
	Env map[string]string

	// HoverKind controls the information that appears in the hover text.
	// SingleLine and Structured are intended for use only by authors of editor plugins.
	HoverKind HoverKind

	// Placeholders enables placeholders for function parameters or struct fields in completion responses.
	UsePlaceholders bool

	// LinkTarget controls where documentation links go.
	// It might be one of:
	//
	// * `"godoc.org"`
	// * `"pkg.go.dev"`
	//
	// If company chooses to use its own `godoc.org`, its address can be used as well.
	LinkTarget string

	// Local is the equivalent of the `goimports -local` flag, which puts imports beginning with this string after 3rd-party packages.
	// It should be the prefix of the import path whose imports should be grouped separately.
	Local string

	// Gofumpt indicates if we should run gofumpt formatting.
	Gofumpt bool

	// Analyses specify analyses that the user would like to enable or disable.
	// A map of the names of analysis passes that should be enabled/disabled.
	// A full list of analyzers that gopls uses can be found [here](analyzers.md)
	//
	// Example Usage:
	// ```json5
	// ...
	// "analyses": {
	//   "unreachable": false, // Disable the unreachable analyzer.
	//   "unusedparams": true  // Enable the unusedparams analyzer.
	// }
	// ...
	// ```
	Analyses map[string]bool

	// Codelenses overrides the enabled/disabled state of code lenses. See the "Code Lenses"
	// section of settings.md for the list of supported lenses.
	//
	// Example Usage:
	// ```json5
	// "gopls": {
	// ...
	//   "codelens": {
	//     "generate": false,  // Don't show the `go generate` lens.
	//     "gc_details": true  // Show a code lens toggling the display of gc's choices.
	//   }
	// ...
	// }
	// ```
	Codelenses map[string]bool

	// LinksInHover toggles the presence of links to documentation in hover.
	LinksInHover bool

	// ImportShortcut specifies whether import statements should link to
	// documentation or go to definitions.
	ImportShortcut ImportShortcut

	// Matcher sets the algorithm that is used when calculating completion candidates.
	Matcher Matcher

	// SymbolMatcher sets the algorithm that is used when finding workspace symbols.
	SymbolMatcher SymbolMatcher

	// SymbolStyle controls how symbols are qualified in symbol responses.
	//
	// Example Usage:
	// ```json5
	// "gopls": {
	// ...
	//   "symbolStyle": "dynamic",
	// ...
	// }
	// ```
	SymbolStyle SymbolStyle
}

// EnvSlice returns Env as a slice of k=v strings.
func (u *UserOptions) EnvSlice() []string {
	var result []string
	for k, v := range u.Env {
		result = append(result, fmt.Sprintf("%v=%v", k, v))
	}
	return result
}

// SetEnvSlice sets Env from a slice of k=v strings.
func (u *UserOptions) SetEnvSlice(env []string) {
	u.Env = map[string]string{}
	for _, kv := range env {
		split := strings.SplitN(kv, "=", 2)
		if len(split) != 2 {
			continue
		}
		u.Env[split[0]] = split[1]
	}
}

// Hooks contains configuration that is provided to the Gopls command by the
// main package.
type Hooks struct {
	GoDiff               bool
	ComputeEdits         diff.ComputeEdits
	URLRegexp            *regexp.Regexp
	GofumptFormat        func(ctx context.Context, src []byte) ([]byte, error)
	DefaultAnalyzers     map[string]Analyzer
	TypeErrorAnalyzers   map[string]Analyzer
	ConvenienceAnalyzers map[string]Analyzer
	StaticcheckAnalyzers map[string]Analyzer
}

// ExperimentalOptions defines configuration for features under active
// development. WARNING: This configuration will be changed in the future. It
// only exists while these features are under development.
type ExperimentalOptions struct {

	// Annotations suppress various kinds of optimization diagnostics
	// that would be reported by the gc_details command.
	//  * noNilcheck suppresses display of nilchecks.
	//  * noEscape suppresses escape choices.
	//  * noInline suppresses inlining choices.
	//  * noBounds suppresses bounds checking diagnostics.
	Annotations map[string]bool

	// Staticcheck enables additional analyses from staticcheck.io.
	Staticcheck bool

	// SemanticTokens controls whether the LSP server will send
	// semantic tokens to the client.
	SemanticTokens bool

	// ExpandWorkspaceToModule instructs `gopls` to adjust the scope of the
	// workspace to find the best available module root. `gopls` first looks for
	// a go.mod file in any parent directory of the workspace folder, expanding
	// the scope to that directory if it exists. If no viable parent directory is
	// found, gopls will check if there is exactly one child directory containing
	// a go.mod file, narrowing the scope to that directory if it exists.
	ExpandWorkspaceToModule bool

	// ExperimentalWorkspaceModule opts a user into the experimental support
	// for multi-module workspaces.
	ExperimentalWorkspaceModule bool

	// ExperimentalDiagnosticsDelay controls the amount of time that gopls waits
	// after the most recent file modification before computing deep diagnostics.
	// Simple diagnostics (parsing and type-checking) are always run immediately
	// on recently modified packages.
	//
	// This option must be set to a valid duration string, for example `"250ms"`.
	ExperimentalDiagnosticsDelay time.Duration

	// ExperimentalPackageCacheKey controls whether to use a coarser cache key
	// for package type information to increase cache hits. This setting removes
	// the user's environment, build flags, and working directory from the cache
	// key, which should be a safe change as all relevant inputs into the type
	// checking pass are already hashed into the key. This is temporarily guarded
	// by an experiment because caching behavior is subtle and difficult to
	// comprehensively test.
	ExperimentalPackageCacheKey bool

	// AllowModfileModifications disables -mod=readonly, allowing imports from
	// out-of-scope modules. This option will eventually be removed.
	AllowModfileModifications bool

	// AllowImplicitNetworkAccess disables GOPROXY=off, allowing implicit module
	// downloads rather than requiring user action. This option will eventually
	// be removed.
	AllowImplicitNetworkAccess bool
}

// DebuggingOptions should not affect the logical execution of Gopls, but may
// be altered for debugging purposes.
type DebuggingOptions struct {
	// VerboseOutput enables additional debug logging.
	VerboseOutput bool

	// CompletionBudget is the soft latency goal for completion requests. Most
	// requests finish in a couple milliseconds, but in some cases deep
	// completions can take much longer. As we use up our budget we
	// dynamically reduce the search scope to ensure we return timely
	// results. Zero means unlimited.
	CompletionBudget time.Duration
}

// InternalOptions contains settings that are not intended for use by the
// average user. These may be settings used by tests or outdated settings that
// will soon be deprecated. Some of these settings may not even be configurable
// by the user.
type InternalOptions struct {
	// LiteralCompletions controls whether literal candidates such as
	// "&someStruct{}" are offered. Tests disable this flag to simplify
	// their expected values.
	LiteralCompletions bool

	// VerboseWorkDoneProgress controls whether the LSP server should send
	// progress reports for all work done outside the scope of an RPC.
	// Used by the regression tests.
	VerboseWorkDoneProgress bool

	// The following options were previously available to users, but they
	// really shouldn't be configured by anyone other than "power users".

	// CompletionDocumentation enables documentation with completion results.
	CompletionDocumentation bool

	// CompleteUnimported enables completion for packages that you do not
	// currently import.
	CompleteUnimported bool

	// DeepCompletion enables the ability to return completions from deep
	// inside relevant entities, rather than just the locally accessible ones.
	//
	// Consider this example:
	//
	// ```go
	// package main
	//
	// import "fmt"
	//
	// type wrapString struct {
	//     str string
	// }
	//
	// func main() {
	//     x := wrapString{"hello world"}
	//     fmt.Printf(<>)
	// }
	// ```
	//
	// At the location of the `<>` in this program, deep completion would suggest the result `x.str`.
	DeepCompletion bool

	// TempModfile controls the use of the -modfile flag in Go 1.14.
	TempModfile bool
}

type ImportShortcut string

const (
	Both       ImportShortcut = "Both"
	Link       ImportShortcut = "Link"
	Definition ImportShortcut = "Definition"
)

func (s ImportShortcut) ShowLinks() bool {
	return s == Both || s == Link
}

func (s ImportShortcut) ShowDefinition() bool {
	return s == Both || s == Definition
}

type Matcher string

const (
	Fuzzy           Matcher = "Fuzzy"
	CaseInsensitive Matcher = "CaseInsensitive"
	CaseSensitive   Matcher = "CaseSensitive"
)

type SymbolMatcher string

const (
	SymbolFuzzy           SymbolMatcher = "Fuzzy"
	SymbolCaseInsensitive SymbolMatcher = "CaseInsensitive"
	SymbolCaseSensitive   SymbolMatcher = "CaseSensitive"
)

type SymbolStyle string

const (
	// PackageQualifiedSymbols is package qualified symbols i.e.
	// "pkg.Foo.Field".
	PackageQualifiedSymbols SymbolStyle = "Package"
	// FullyQualifiedSymbols is fully qualified symbols, i.e.
	// "path/to/pkg.Foo.Field".
	FullyQualifiedSymbols SymbolStyle = "Full"
	// DynamicSymbols uses whichever qualifier results in the highest scoring
	// match for the given symbol query. Here a "qualifier" is any "/" or "."
	// delimited suffix of the fully qualified symbol. i.e. "to/pkg.Foo.Field" or
	// just "Foo.Field".
	DynamicSymbols SymbolStyle = "Dynamic"
)

type HoverKind string

const (
	SingleLine            HoverKind = "SingleLine"
	NoDocumentation       HoverKind = "NoDocumentation"
	SynopsisDocumentation HoverKind = "SynopsisDocumentation"
	FullDocumentation     HoverKind = "FullDocumentation"

	// Structured is an experimental setting that returns a structured hover format.
	// This format separates the signature from the documentation, so that the client
	// can do more manipulation of these fields.
	//
	// This should only be used by clients that support this behavior.
	Structured HoverKind = "Structured"
)

type OptionResults []OptionResult

type OptionResult struct {
	Name  string
	Value interface{}
	Error error

	State       OptionState
	Replacement string
}

type OptionState int

const (
	OptionHandled = OptionState(iota)
	OptionDeprecated
	OptionUnexpected
)

type LinkTarget string

func SetOptions(options *Options, opts interface{}) OptionResults {
	var results OptionResults
	switch opts := opts.(type) {
	case nil:
	case map[string]interface{}:
		// If the user's settings contains "allExperiments", set that first,
		// and then let them override individual settings independently.
		var enableExperiments bool
		for name, value := range opts {
			if b, ok := value.(bool); name == "allExperiments" && ok && b {
				enableExperiments = true
				options.enableAllExperiments()
			}
		}
		for name, value := range opts {
			results = append(results, options.set(name, value))
		}
		// Finally, enable any experimental features that are specified in
		// maps, which allows users to individually toggle them on or off.
		if enableExperiments {
			options.enableAllExperimentMaps()
		}
	default:
		results = append(results, OptionResult{
			Value: opts,
			Error: errors.Errorf("Invalid options type %T", opts),
		})
	}
	return results
}

func (o *Options) ForClientCapabilities(caps protocol.ClientCapabilities) {
	// Check if the client supports snippets in completion items.
	if c := caps.TextDocument.Completion; c.CompletionItem.SnippetSupport {
		o.InsertTextFormat = protocol.SnippetTextFormat
	}
	// Check if the client supports configuration messages.
	o.ConfigurationSupported = caps.Workspace.Configuration
	o.DynamicConfigurationSupported = caps.Workspace.DidChangeConfiguration.DynamicRegistration
	o.DynamicWatchedFilesSupported = caps.Workspace.DidChangeWatchedFiles.DynamicRegistration

	// Check which types of content format are supported by this client.
	if hover := caps.TextDocument.Hover; len(hover.ContentFormat) > 0 {
		o.PreferredContentFormat = hover.ContentFormat[0]
	}
	// Check if the client supports only line folding.
	fr := caps.TextDocument.FoldingRange
	o.LineFoldingOnly = fr.LineFoldingOnly
	// Check if the client supports hierarchical document symbols.
	o.HierarchicalDocumentSymbolSupport = caps.TextDocument.DocumentSymbol.HierarchicalDocumentSymbolSupport
	// Check if the client supports semantic tokens
	o.SemanticTypes = caps.TextDocument.SemanticTokens.TokenTypes
	o.SemanticMods = caps.TextDocument.SemanticTokens.TokenModifiers
	// we don't need Requests, as we support full functionality
	// we don't need Formats, as there is only one, for now

}

func (o *Options) Clone() *Options {
	result := &Options{
		ClientOptions:       o.ClientOptions,
		DebuggingOptions:    o.DebuggingOptions,
		ExperimentalOptions: o.ExperimentalOptions,
		InternalOptions:     o.InternalOptions,
		Hooks: Hooks{
			GoDiff:        o.Hooks.GoDiff,
			ComputeEdits:  o.Hooks.ComputeEdits,
			GofumptFormat: o.GofumptFormat,
			URLRegexp:     o.URLRegexp,
		},
		ServerOptions: o.ServerOptions,
		UserOptions:   o.UserOptions,
	}
	// Fully clone any slice or map fields. Only Hooks, ExperimentalOptions,
	// and UserOptions can be modified.
	copyStringMap := func(src map[string]bool) map[string]bool {
		dst := make(map[string]bool)
		for k, v := range src {
			dst[k] = v
		}
		return dst
	}
	result.Analyses = copyStringMap(o.Analyses)
	result.Annotations = copyStringMap(o.Annotations)
	result.Codelenses = copyStringMap(o.Codelenses)

	copySlice := func(src []string) []string {
		dst := make([]string, len(src))
		copy(dst, src)
		return dst
	}
	result.SetEnvSlice(o.EnvSlice())
	result.BuildFlags = copySlice(o.BuildFlags)

	copyAnalyzerMap := func(src map[string]Analyzer) map[string]Analyzer {
		dst := make(map[string]Analyzer)
		for k, v := range src {
			dst[k] = v
		}
		return dst
	}
	result.DefaultAnalyzers = copyAnalyzerMap(o.DefaultAnalyzers)
	result.TypeErrorAnalyzers = copyAnalyzerMap(o.TypeErrorAnalyzers)
	result.ConvenienceAnalyzers = copyAnalyzerMap(o.ConvenienceAnalyzers)
	result.StaticcheckAnalyzers = copyAnalyzerMap(o.StaticcheckAnalyzers)
	return result
}

func (o *Options) AddStaticcheckAnalyzer(a *analysis.Analyzer) {
	o.StaticcheckAnalyzers[a.Name] = Analyzer{Analyzer: a, Enabled: true}
}

// enableAllExperiments turns on all of the experimental "off-by-default"
// features offered by gopls.
// Any experimental features specified in maps should be enabled in
// enableAllExperimentMaps.
func (o *Options) enableAllExperiments() {
	o.ExperimentalDiagnosticsDelay = 200 * time.Millisecond
}

func (o *Options) enableAllExperimentMaps() {
	if _, ok := o.Codelenses[CommandToggleDetails.Name]; !ok {
		o.Codelenses[CommandToggleDetails.Name] = true
	}
	if _, ok := o.Analyses[unusedparams.Analyzer.Name]; !ok {
		o.Analyses[unusedparams.Analyzer.Name] = true
	}
}

func (o *Options) set(name string, value interface{}) OptionResult {
	result := OptionResult{Name: name, Value: value}
	switch name {
	case "env":
		menv, ok := value.(map[string]interface{})
		if !ok {
			result.errorf("invalid type %T, expect map", value)
			break
		}
		for k, v := range menv {
			o.Env[k] = fmt.Sprint(v)
		}

	case "buildFlags":
		iflags, ok := value.([]interface{})
		if !ok {
			result.errorf("invalid type %T, expect list", value)
			break
		}
		flags := make([]string, 0, len(iflags))
		for _, flag := range iflags {
			flags = append(flags, fmt.Sprintf("%s", flag))
		}
		o.BuildFlags = flags

	case "completionDocumentation":
		result.setBool(&o.CompletionDocumentation)
	case "usePlaceholders":
		result.setBool(&o.UsePlaceholders)
	case "deepCompletion":
		result.setBool(&o.DeepCompletion)
	case "completeUnimported":
		result.setBool(&o.CompleteUnimported)
	case "completionBudget":
		result.setDuration(&o.CompletionBudget)
	case "matcher":
		if s, ok := result.asOneOf(
			string(Fuzzy),
			string(CaseSensitive),
			string(CaseInsensitive),
		); ok {
			o.Matcher = Matcher(s)
		}

	case "symbolMatcher":
		if s, ok := result.asOneOf(
			string(SymbolFuzzy),
			string(SymbolCaseInsensitive),
			string(SymbolCaseSensitive),
		); ok {
			o.SymbolMatcher = SymbolMatcher(s)
		}

	case "symbolStyle":
		if s, ok := result.asOneOf(
			string(FullyQualifiedSymbols),
			string(PackageQualifiedSymbols),
			string(DynamicSymbols),
		); ok {
			o.SymbolStyle = SymbolStyle(s)
		}

	case "hoverKind":
		if s, ok := result.asOneOf(
			string(NoDocumentation),
			string(SingleLine),
			string(SynopsisDocumentation),
			string(FullDocumentation),
			string(Structured),
		); ok {
			o.HoverKind = HoverKind(s)
		}

	case "linkTarget":
		result.setString(&o.LinkTarget)

	case "linksInHover":
		result.setBool(&o.LinksInHover)

	case "importShortcut":
		if s, ok := result.asOneOf(string(Both), string(Link), string(Definition)); ok {
			o.ImportShortcut = ImportShortcut(s)
		}

	case "analyses":
		result.setBoolMap(&o.Analyses)

	case "annotations":
		result.setBoolMap(&o.Annotations)
		for k := range o.Annotations {
			switch k {
			case "noEscape", "noNilcheck", "noInline", "noBounds":
				continue
			default:
				result.Name += ":" + k // put mistake(s) in the message
				result.State = OptionUnexpected
			}
		}

	case "codelenses", "codelens":
		var lensOverrides map[string]bool
		result.setBoolMap(&lensOverrides)
		if result.Error == nil {
			if o.Codelenses == nil {
				o.Codelenses = make(map[string]bool)
			}
			for lens, enabled := range lensOverrides {
				o.Codelenses[lens] = enabled
			}
		}

		// codelens is deprecated, but still works for now.
		// TODO(rstambler): Remove this for the gopls/v0.7.0 release.
		if name == "codelens" {
			result.State = OptionDeprecated
			result.Replacement = "codelenses"
		}

	case "staticcheck":
		result.setBool(&o.Staticcheck)

	case "local":
		result.setString(&o.Local)

	case "verboseOutput":
		result.setBool(&o.VerboseOutput)

	case "verboseWorkDoneProgress":
		result.setBool(&o.VerboseWorkDoneProgress)

	case "tempModfile":
		result.setBool(&o.TempModfile)

	case "gofumpt":
		result.setBool(&o.Gofumpt)

	case "semanticTokens":
		result.setBool(&o.SemanticTokens)

	case "expandWorkspaceToModule":
		result.setBool(&o.ExpandWorkspaceToModule)

	case "experimentalWorkspaceModule":
		result.setBool(&o.ExperimentalWorkspaceModule)

	case "experimentalDiagnosticsDelay":
		result.setDuration(&o.ExperimentalDiagnosticsDelay)

	case "experimentalPackageCacheKey":
		result.setBool(&o.ExperimentalPackageCacheKey)

	case "allExperiments":
		// This setting should be handled before all of the other options are
		// processed, so do nothing here.

	// Replaced settings.
	case "experimentalDisabledAnalyses":
		result.State = OptionDeprecated
		result.Replacement = "analyses"

	case "disableDeepCompletion":
		result.State = OptionDeprecated
		result.Replacement = "deepCompletion"

	case "disableFuzzyMatching":
		result.State = OptionDeprecated
		result.Replacement = "fuzzyMatching"

	case "wantCompletionDocumentation":
		result.State = OptionDeprecated
		result.Replacement = "completionDocumentation"

	case "wantUnimportedCompletions":
		result.State = OptionDeprecated
		result.Replacement = "completeUnimported"

	case "fuzzyMatching":
		result.State = OptionDeprecated
		result.Replacement = "matcher"

	case "caseSensitiveCompletion":
		result.State = OptionDeprecated
		result.Replacement = "matcher"

	// Deprecated settings.
	case "wantSuggestedFixes":
		result.State = OptionDeprecated

	case "noIncrementalSync":
		result.State = OptionDeprecated

	case "watchFileChanges":
		result.State = OptionDeprecated

	case "go-diff":
		result.State = OptionDeprecated

	default:
		result.State = OptionUnexpected
	}
	return result
}

func (r *OptionResult) errorf(msg string, values ...interface{}) {
	prefix := fmt.Sprintf("parsing setting %q: ", r.Name)
	r.Error = errors.Errorf(prefix+msg, values...)
}

func (r *OptionResult) asBool() (bool, bool) {
	b, ok := r.Value.(bool)
	if !ok {
		r.errorf("invalid type %T, expect bool", r.Value)
		return false, false
	}
	return b, true
}

func (r *OptionResult) setBool(b *bool) {
	if v, ok := r.asBool(); ok {
		*b = v
	}
}

func (r *OptionResult) setDuration(d *time.Duration) {
	if v, ok := r.asString(); ok {
		parsed, err := time.ParseDuration(v)
		if err != nil {
			r.errorf("failed to parse duration %q: %v", v, err)
			return
		}
		*d = parsed
	}
}

func (r *OptionResult) setBoolMap(bm *map[string]bool) {
	all, ok := r.Value.(map[string]interface{})
	if !ok {
		r.errorf("invalid type %T for map[string]bool option", r.Value)
		return
	}
	m := make(map[string]bool)
	for a, enabled := range all {
		if enabled, ok := enabled.(bool); ok {
			m[a] = enabled
		} else {
			r.errorf("invalid type %T for map key %q", enabled, a)
			return
		}
	}
	*bm = m
}

func (r *OptionResult) asString() (string, bool) {
	b, ok := r.Value.(string)
	if !ok {
		r.errorf("invalid type %T, expect string", r.Value)
		return "", false
	}
	return b, true
}

func (r *OptionResult) asOneOf(options ...string) (string, bool) {
	s, ok := r.asString()
	if !ok {
		return "", false
	}
	lower := strings.ToLower(s)
	for _, opt := range options {
		if strings.ToLower(opt) == lower {
			return opt, true
		}
	}
	r.errorf("invalid option %q for enum", r.Value)
	return "", false
}

func (r *OptionResult) setString(s *string) {
	if v, ok := r.asString(); ok {
		*s = v
	}
}

// EnabledAnalyzers returns all of the analyzers enabled for the given
// snapshot.
func EnabledAnalyzers(snapshot Snapshot) (analyzers []Analyzer) {
	for _, a := range snapshot.View().Options().DefaultAnalyzers {
		if a.IsEnabled(snapshot.View()) {
			analyzers = append(analyzers, a)
		}
	}
	for _, a := range snapshot.View().Options().TypeErrorAnalyzers {
		if a.IsEnabled(snapshot.View()) {
			analyzers = append(analyzers, a)
		}
	}
	for _, a := range snapshot.View().Options().ConvenienceAnalyzers {
		if a.IsEnabled(snapshot.View()) {
			analyzers = append(analyzers, a)
		}
	}
	for _, a := range snapshot.View().Options().StaticcheckAnalyzers {
		if a.IsEnabled(snapshot.View()) {
			analyzers = append(analyzers, a)
		}
	}
	return analyzers
}

func typeErrorAnalyzers() map[string]Analyzer {
	return map[string]Analyzer{
		fillreturns.Analyzer.Name: {
			Analyzer:       fillreturns.Analyzer,
			FixesError:     fillreturns.FixesError,
			HighConfidence: true,
			Enabled:        true,
		},
		nonewvars.Analyzer.Name: {
			Analyzer:   nonewvars.Analyzer,
			FixesError: nonewvars.FixesError,
			Enabled:    true,
		},
		noresultvalues.Analyzer.Name: {
			Analyzer:   noresultvalues.Analyzer,
			FixesError: noresultvalues.FixesError,
			Enabled:    true,
		},
		undeclaredname.Analyzer.Name: {
			Analyzer:   undeclaredname.Analyzer,
			FixesError: undeclaredname.FixesError,
			Command:    CommandUndeclaredName,
			Enabled:    true,
		},
	}
}

func convenienceAnalyzers() map[string]Analyzer {
	return map[string]Analyzer{
		fillstruct.Analyzer.Name: {
			Analyzer: fillstruct.Analyzer,
			Command:  CommandFillStruct,
			Enabled:  true,
		},
	}
}

func defaultAnalyzers() map[string]Analyzer {
	return map[string]Analyzer{
		// The traditional vet suite:
		asmdecl.Analyzer.Name:       {Analyzer: asmdecl.Analyzer, Enabled: true},
		assign.Analyzer.Name:        {Analyzer: assign.Analyzer, Enabled: true},
		atomic.Analyzer.Name:        {Analyzer: atomic.Analyzer, Enabled: true},
		bools.Analyzer.Name:         {Analyzer: bools.Analyzer, Enabled: true},
		buildtag.Analyzer.Name:      {Analyzer: buildtag.Analyzer, Enabled: true},
		cgocall.Analyzer.Name:       {Analyzer: cgocall.Analyzer, Enabled: true},
		composite.Analyzer.Name:     {Analyzer: composite.Analyzer, Enabled: true},
		copylock.Analyzer.Name:      {Analyzer: copylock.Analyzer, Enabled: true},
		errorsas.Analyzer.Name:      {Analyzer: errorsas.Analyzer, Enabled: true},
		httpresponse.Analyzer.Name:  {Analyzer: httpresponse.Analyzer, Enabled: true},
		ifaceassert.Analyzer.Name:   {Analyzer: ifaceassert.Analyzer, Enabled: true},
		loopclosure.Analyzer.Name:   {Analyzer: loopclosure.Analyzer, Enabled: true},
		lostcancel.Analyzer.Name:    {Analyzer: lostcancel.Analyzer, Enabled: true},
		nilfunc.Analyzer.Name:       {Analyzer: nilfunc.Analyzer, Enabled: true},
		printf.Analyzer.Name:        {Analyzer: printf.Analyzer, Enabled: true},
		shift.Analyzer.Name:         {Analyzer: shift.Analyzer, Enabled: true},
		stdmethods.Analyzer.Name:    {Analyzer: stdmethods.Analyzer, Enabled: true},
		stringintconv.Analyzer.Name: {Analyzer: stringintconv.Analyzer, Enabled: true},
		structtag.Analyzer.Name:     {Analyzer: structtag.Analyzer, Enabled: true},
		tests.Analyzer.Name:         {Analyzer: tests.Analyzer, Enabled: true},
		unmarshal.Analyzer.Name:     {Analyzer: unmarshal.Analyzer, Enabled: true},
		unreachable.Analyzer.Name:   {Analyzer: unreachable.Analyzer, Enabled: true},
		unsafeptr.Analyzer.Name:     {Analyzer: unsafeptr.Analyzer, Enabled: true},
		unusedresult.Analyzer.Name:  {Analyzer: unusedresult.Analyzer, Enabled: true},

		// Non-vet analyzers:
		atomicalign.Analyzer.Name:      {Analyzer: atomicalign.Analyzer, Enabled: true},
		deepequalerrors.Analyzer.Name:  {Analyzer: deepequalerrors.Analyzer, Enabled: true},
		fieldalignment.Analyzer.Name:   {Analyzer: fieldalignment.Analyzer, Enabled: false},
		sortslice.Analyzer.Name:        {Analyzer: sortslice.Analyzer, Enabled: true},
		testinggoroutine.Analyzer.Name: {Analyzer: testinggoroutine.Analyzer, Enabled: true},
		unusedparams.Analyzer.Name:     {Analyzer: unusedparams.Analyzer, Enabled: false},

		// gofmt -s suite:
		simplifycompositelit.Analyzer.Name: {Analyzer: simplifycompositelit.Analyzer, Enabled: true, HighConfidence: true},
		simplifyrange.Analyzer.Name:        {Analyzer: simplifyrange.Analyzer, Enabled: true, HighConfidence: true},
		simplifyslice.Analyzer.Name:        {Analyzer: simplifyslice.Analyzer, Enabled: true, HighConfidence: true},
	}
}

func urlRegexp() *regexp.Regexp {
	// Ensure links are matched as full words, not anywhere.
	re := regexp.MustCompile(`\b(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?\b`)
	re.Longest()
	return re
}

type APIJSON struct {
	Options  map[string][]*OptionJSON
	Commands []*CommandJSON
	Lenses   []*LensJSON
}

type OptionJSON struct {
	Name       string
	Type       string
	Doc        string
	EnumValues []EnumValue
	Default    string
}

type EnumValue struct {
	Value string
	Doc   string
}

type CommandJSON struct {
	Command string
	Title   string
	Doc     string
}

type LensJSON struct {
	Lens  string
	Title string
	Doc   string
}
