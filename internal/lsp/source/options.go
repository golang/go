// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"fmt"
	"os"
	"regexp"
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
	"golang.org/x/tools/go/analysis/passes/httpresponse"
	"golang.org/x/tools/go/analysis/passes/loopclosure"
	"golang.org/x/tools/go/analysis/passes/lostcancel"
	"golang.org/x/tools/go/analysis/passes/nilfunc"
	"golang.org/x/tools/go/analysis/passes/printf"
	"golang.org/x/tools/go/analysis/passes/shift"
	"golang.org/x/tools/go/analysis/passes/sortslice"
	"golang.org/x/tools/go/analysis/passes/stdmethods"
	"golang.org/x/tools/go/analysis/passes/structtag"
	"golang.org/x/tools/go/analysis/passes/testinggoroutine"
	"golang.org/x/tools/go/analysis/passes/tests"
	"golang.org/x/tools/go/analysis/passes/unmarshal"
	"golang.org/x/tools/go/analysis/passes/unreachable"
	"golang.org/x/tools/go/analysis/passes/unsafeptr"
	"golang.org/x/tools/go/analysis/passes/unusedresult"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/diff/myers"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/telemetry/tag"
	errors "golang.org/x/xerrors"
)

func DefaultOptions() Options {
	return Options{
		ClientOptions: ClientOptions{
			InsertTextFormat:              protocol.PlainTextTextFormat,
			PreferredContentFormat:        protocol.Markdown,
			ConfigurationSupported:        true,
			DynamicConfigurationSupported: true,
			DynamicWatchedFilesSupported:  true,
			LineFoldingOnly:               false,
		},
		ServerOptions: ServerOptions{
			SupportedCodeActions: map[FileKind]map[protocol.CodeActionKind]bool{
				Go: {
					protocol.SourceOrganizeImports: true,
					protocol.QuickFix:              true,
				},
				Mod: {
					protocol.SourceOrganizeImports: true,
				},
				Sum: {},
			},
			SupportedCommands: []string{
				"tidy",               // for go.mod files
				"upgrade.dependency", // for go.mod dependency upgrades
			},
		},
		UserOptions: UserOptions{
			Env:                     os.Environ(),
			HoverKind:               FullDocumentation,
			LinkTarget:              "pkg.go.dev",
			Matcher:                 Fuzzy,
			DeepCompletion:          true,
			UnimportedCompletion:    true,
			CompletionDocumentation: true,
		},
		DebuggingOptions: DebuggingOptions{
			CompletionBudget: 100 * time.Millisecond,
		},
		ExperimentalOptions: ExperimentalOptions{
			TempModfile: true,
		},
		Hooks: Hooks{
			ComputeEdits: myers.ComputeEdits,
			URLRegexp:    regexp.MustCompile(`(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?`),
			Analyzers:    defaultAnalyzers(),
			GoDiff:       true,
		},
	}
}

type Options struct {
	ClientOptions
	ServerOptions
	UserOptions
	DebuggingOptions
	ExperimentalOptions
	Hooks
}

type ClientOptions struct {
	InsertTextFormat              protocol.InsertTextFormat
	ConfigurationSupported        bool
	DynamicConfigurationSupported bool
	DynamicWatchedFilesSupported  bool
	PreferredContentFormat        protocol.MarkupKind
	LineFoldingOnly               bool
}

type ServerOptions struct {
	SupportedCodeActions map[FileKind]map[protocol.CodeActionKind]bool
	SupportedCommands    []string
}

type UserOptions struct {
	// Env is the current set of environment overrides on this view.
	Env []string

	// BuildFlags is used to adjust the build flags applied to the view.
	BuildFlags []string

	// HoverKind specifies the format of the content for hover requests.
	HoverKind HoverKind

	// DisabledAnalyses specify analyses that the user would like to disable.
	DisabledAnalyses map[string]struct{}

	// StaticCheck enables additional analyses from staticcheck.io.
	StaticCheck bool

	// LinkTarget is the website used for documentation.
	LinkTarget string

	// LocalPrefix is used to specify goimports's -local behavior.
	LocalPrefix string

	// Matcher specifies the type of matcher to use for completion requests.
	Matcher Matcher

	// DeepCompletion allows completion to perform nested searches through
	// possible candidates.
	DeepCompletion bool

	// UnimportedCompletion enables completion for unimported packages.
	UnimportedCompletion bool

	// CompletionDocumentation returns additional documentation with completion
	// requests.
	CompletionDocumentation bool

	// Placeholders adds placeholders to parameters and structs in completion
	// results.
	Placeholders bool
}

type completionOptions struct {
	deepCompletion    bool
	unimported        bool
	documentation     bool
	fullDocumentation bool
	placeholders      bool
	literal           bool
	matcher           Matcher
	budget            time.Duration
}

type Hooks struct {
	GoDiff       bool
	ComputeEdits diff.ComputeEdits
	URLRegexp    *regexp.Regexp
	Analyzers    map[string]*analysis.Analyzer
}

type ExperimentalOptions struct {
	// WARNING: This configuration will be changed in the future.
	// It only exists while this feature is under development.
	// Disable use of the -modfile flag in Go 1.14.
	TempModfile bool
}

type DebuggingOptions struct {
	VerboseOutput bool

	// CompletionBudget is the soft latency goal for completion requests. Most
	// requests finish in a couple milliseconds, but in some cases deep
	// completions can take much longer. As we use up our budget we
	// dynamically reduce the search scope to ensure we return timely
	// results. Zero means unlimited.
	CompletionBudget time.Duration
}

type Matcher int

const (
	Fuzzy = Matcher(iota)
	CaseInsensitive
	CaseSensitive
)

type HoverKind int

const (
	SingleLine = HoverKind(iota)
	NoDocumentation
	SynopsisDocumentation
	FullDocumentation

	// Structured is an experimental setting that returns a structured hover format.
	// This format separates the signature from the documentation, so that the client
	// can do more manipulation of these fields.
	//
	// This should only be used by clients that support this behavior.
	Structured
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
		for name, value := range opts {
			results = append(results, options.set(name, value))
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
}

func (o *Options) set(name string, value interface{}) OptionResult {
	result := OptionResult{Name: name, Value: value}
	switch name {
	case "env":
		menv, ok := value.(map[string]interface{})
		if !ok {
			result.errorf("invalid config gopls.env type %T", value)
			break
		}
		for k, v := range menv {
			o.Env = append(o.Env, fmt.Sprintf("%s=%s", k, v))
		}

	case "buildFlags":
		iflags, ok := value.([]interface{})
		if !ok {
			result.errorf("invalid config gopls.buildFlags type %T", value)
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
		result.setBool(&o.Placeholders)
	case "deepCompletion":
		result.setBool(&o.DeepCompletion)
	case "completeUnimported":
		result.setBool(&o.UnimportedCompletion)
	case "completionBudget":
		if v, ok := result.asString(); ok {
			d, err := time.ParseDuration(v)
			if err != nil {
				result.errorf("failed to parse duration %q: %v", v, err)
				break
			}
			o.CompletionBudget = d
		}

	case "matcher":
		matcher, ok := result.asString()
		if !ok {
			break
		}
		switch matcher {
		case "fuzzy":
			o.Matcher = Fuzzy
		case "caseSensitive":
			o.Matcher = CaseSensitive
		default:
			o.Matcher = CaseInsensitive
		}

	case "hoverKind":
		hoverKind, ok := result.asString()
		if !ok {
			break
		}
		switch hoverKind {
		case "NoDocumentation":
			o.HoverKind = NoDocumentation
		case "SingleLine":
			o.HoverKind = SingleLine
		case "SynopsisDocumentation":
			o.HoverKind = SynopsisDocumentation
		case "FullDocumentation":
			o.HoverKind = FullDocumentation
		case "Structured":
			o.HoverKind = Structured
		default:
			result.errorf("Unsupported hover kind", tag.Of("HoverKind", hoverKind))
		}

	case "linkTarget":
		linkTarget, ok := value.(string)
		if !ok {
			result.errorf("invalid type %T for string option %q", value, name)
			break
		}
		o.LinkTarget = linkTarget

	case "experimentalDisabledAnalyses":
		disabledAnalyses, ok := value.([]interface{})
		if !ok {
			result.errorf("Invalid type %T for []string option %q", value, name)
			break
		}
		o.DisabledAnalyses = make(map[string]struct{})
		for _, a := range disabledAnalyses {
			o.DisabledAnalyses[fmt.Sprint(a)] = struct{}{}
		}

	case "staticcheck":
		result.setBool(&o.StaticCheck)

	case "local":
		localPrefix, ok := value.(string)
		if !ok {
			result.errorf("invalid type %T for string option %q", value, name)
			break
		}
		o.LocalPrefix = localPrefix

	case "verboseOutput":
		result.setBool(&o.VerboseOutput)

	case "tempModfile":
		result.setBool(&o.TempModfile)

	// Deprecated settings.
	case "wantSuggestedFixes":
		result.State = OptionDeprecated

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
	r.Error = errors.Errorf(msg, values...)
}

func (r *OptionResult) asBool() (bool, bool) {
	b, ok := r.Value.(bool)
	if !ok {
		r.errorf("Invalid type %T for bool option %q", r.Value, r.Name)
		return false, false
	}
	return b, true
}

func (r *OptionResult) asString() (string, bool) {
	b, ok := r.Value.(string)
	if !ok {
		r.errorf("Invalid type %T for string option %q", r.Value, r.Name)
		return "", false
	}
	return b, true
}

func (r *OptionResult) setBool(b *bool) {
	if v, ok := r.asBool(); ok {
		*b = v
	}
}

func defaultAnalyzers() map[string]*analysis.Analyzer {
	return map[string]*analysis.Analyzer{
		// The traditional vet suite:
		asmdecl.Analyzer.Name:      asmdecl.Analyzer,
		assign.Analyzer.Name:       assign.Analyzer,
		atomic.Analyzer.Name:       atomic.Analyzer,
		atomicalign.Analyzer.Name:  atomicalign.Analyzer,
		bools.Analyzer.Name:        bools.Analyzer,
		buildtag.Analyzer.Name:     buildtag.Analyzer,
		cgocall.Analyzer.Name:      cgocall.Analyzer,
		composite.Analyzer.Name:    composite.Analyzer,
		copylock.Analyzer.Name:     copylock.Analyzer,
		errorsas.Analyzer.Name:     errorsas.Analyzer,
		httpresponse.Analyzer.Name: httpresponse.Analyzer,
		loopclosure.Analyzer.Name:  loopclosure.Analyzer,
		lostcancel.Analyzer.Name:   lostcancel.Analyzer,
		nilfunc.Analyzer.Name:      nilfunc.Analyzer,
		printf.Analyzer.Name:       printf.Analyzer,
		shift.Analyzer.Name:        shift.Analyzer,
		stdmethods.Analyzer.Name:   stdmethods.Analyzer,
		structtag.Analyzer.Name:    structtag.Analyzer,
		tests.Analyzer.Name:        tests.Analyzer,
		unmarshal.Analyzer.Name:    unmarshal.Analyzer,
		unreachable.Analyzer.Name:  unreachable.Analyzer,
		unsafeptr.Analyzer.Name:    unsafeptr.Analyzer,
		unusedresult.Analyzer.Name: unusedresult.Analyzer,

		// Non-vet analyzers
		deepequalerrors.Analyzer.Name:  deepequalerrors.Analyzer,
		sortslice.Analyzer.Name:        sortslice.Analyzer,
		testinggoroutine.Analyzer.Name: testinggoroutine.Analyzer,
	}
}
