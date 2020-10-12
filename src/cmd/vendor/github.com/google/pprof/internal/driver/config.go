package driver

import (
	"fmt"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"sync"
)

// config holds settings for a single named config.
// The JSON tag name for a field is used both for JSON encoding and as
// a named variable.
type config struct {
	// Filename for file-based output formats, stdout by default.
	Output string `json:"-"`

	// Display options.
	CallTree            bool    `json:"call_tree,omitempty"`
	RelativePercentages bool    `json:"relative_percentages,omitempty"`
	Unit                string  `json:"unit,omitempty"`
	CompactLabels       bool    `json:"compact_labels,omitempty"`
	SourcePath          string  `json:"-"`
	TrimPath            string  `json:"-"`
	IntelSyntax         bool    `json:"intel_syntax,omitempty"`
	Mean                bool    `json:"mean,omitempty"`
	SampleIndex         string  `json:"-"`
	DivideBy            float64 `json:"-"`
	Normalize           bool    `json:"normalize,omitempty"`
	Sort                string  `json:"sort,omitempty"`

	// Filtering options
	DropNegative bool    `json:"drop_negative,omitempty"`
	NodeCount    int     `json:"nodecount,omitempty"`
	NodeFraction float64 `json:"nodefraction,omitempty"`
	EdgeFraction float64 `json:"edgefraction,omitempty"`
	Trim         bool    `json:"trim,omitempty"`
	Focus        string  `json:"focus,omitempty"`
	Ignore       string  `json:"ignore,omitempty"`
	PruneFrom    string  `json:"prune_from,omitempty"`
	Hide         string  `json:"hide,omitempty"`
	Show         string  `json:"show,omitempty"`
	ShowFrom     string  `json:"show_from,omitempty"`
	TagFocus     string  `json:"tagfocus,omitempty"`
	TagIgnore    string  `json:"tagignore,omitempty"`
	TagShow      string  `json:"tagshow,omitempty"`
	TagHide      string  `json:"taghide,omitempty"`
	NoInlines    bool    `json:"noinlines,omitempty"`

	// Output granularity
	Granularity string `json:"granularity,omitempty"`
}

// defaultConfig returns the default configuration values; it is unaffected by
// flags and interactive assignments.
func defaultConfig() config {
	return config{
		Unit:         "minimum",
		NodeCount:    -1,
		NodeFraction: 0.005,
		EdgeFraction: 0.001,
		Trim:         true,
		DivideBy:     1.0,
		Sort:         "flat",
		Granularity:  "functions",
	}
}

// currentConfig holds the current configuration values; it is affected by
// flags and interactive assignments.
var currentCfg = defaultConfig()
var currentMu sync.Mutex

func currentConfig() config {
	currentMu.Lock()
	defer currentMu.Unlock()
	return currentCfg
}

func setCurrentConfig(cfg config) {
	currentMu.Lock()
	defer currentMu.Unlock()
	currentCfg = cfg
}

// configField contains metadata for a single configuration field.
type configField struct {
	name         string              // JSON field name/key in variables
	urlparam     string              // URL parameter name
	saved        bool                // Is field saved in settings?
	field        reflect.StructField // Field in config
	choices      []string            // Name Of variables in group
	defaultValue string              // Default value for this field.
}

var (
	configFields []configField // Precomputed metadata per config field

	// configFieldMap holds an entry for every config field as well as an
	// entry for every valid choice for a multi-choice field.
	configFieldMap map[string]configField
)

func init() {
	// Config names for fields that are not saved in settings and therefore
	// do not have a JSON name.
	notSaved := map[string]string{
		// Not saved in settings, but present in URLs.
		"SampleIndex": "sample_index",

		// Following fields are also not placed in URLs.
		"Output":     "output",
		"SourcePath": "source_path",
		"TrimPath":   "trim_path",
		"DivideBy":   "divide_by",
	}

	// choices holds the list of allowed values for config fields that can
	// take on one of a bounded set of values.
	choices := map[string][]string{
		"sort":        {"cum", "flat"},
		"granularity": {"functions", "filefunctions", "files", "lines", "addresses"},
	}

	// urlparam holds the mapping from a config field name to the URL
	// parameter used to hold that config field. If no entry is present for
	// a name, the corresponding field is not saved in URLs.
	urlparam := map[string]string{
		"drop_negative":        "dropneg",
		"call_tree":            "calltree",
		"relative_percentages": "rel",
		"unit":                 "unit",
		"compact_labels":       "compact",
		"intel_syntax":         "intel",
		"nodecount":            "n",
		"nodefraction":         "nf",
		"edgefraction":         "ef",
		"trim":                 "trim",
		"focus":                "f",
		"ignore":               "i",
		"prune_from":           "prunefrom",
		"hide":                 "h",
		"show":                 "s",
		"show_from":            "sf",
		"tagfocus":             "tf",
		"tagignore":            "ti",
		"tagshow":              "ts",
		"taghide":              "th",
		"mean":                 "mean",
		"sample_index":         "si",
		"normalize":            "norm",
		"sort":                 "sort",
		"granularity":          "g",
		"noinlines":            "noinlines",
	}

	def := defaultConfig()
	configFieldMap = map[string]configField{}
	t := reflect.TypeOf(config{})
	for i, n := 0, t.NumField(); i < n; i++ {
		field := t.Field(i)
		js := strings.Split(field.Tag.Get("json"), ",")
		if len(js) == 0 {
			continue
		}
		// Get the configuration name for this field.
		name := js[0]
		if name == "-" {
			name = notSaved[field.Name]
			if name == "" {
				// Not a configurable field.
				continue
			}
		}
		f := configField{
			name:     name,
			urlparam: urlparam[name],
			saved:    (name == js[0]),
			field:    field,
			choices:  choices[name],
		}
		f.defaultValue = def.get(f)
		configFields = append(configFields, f)
		configFieldMap[f.name] = f
		for _, choice := range f.choices {
			configFieldMap[choice] = f
		}
	}
}

// fieldPtr returns a pointer to the field identified by f in *cfg.
func (cfg *config) fieldPtr(f configField) interface{} {
	// reflect.ValueOf: converts to reflect.Value
	// Elem: dereferences cfg to make *cfg
	// FieldByIndex: fetches the field
	// Addr: takes address of field
	// Interface: converts back from reflect.Value to a regular value
	return reflect.ValueOf(cfg).Elem().FieldByIndex(f.field.Index).Addr().Interface()
}

// get returns the value of field f in cfg.
func (cfg *config) get(f configField) string {
	switch ptr := cfg.fieldPtr(f).(type) {
	case *string:
		return *ptr
	case *int:
		return fmt.Sprint(*ptr)
	case *float64:
		return fmt.Sprint(*ptr)
	case *bool:
		return fmt.Sprint(*ptr)
	}
	panic(fmt.Sprintf("unsupported config field type %v", f.field.Type))
}

// set sets the value of field f in cfg to value.
func (cfg *config) set(f configField, value string) error {
	switch ptr := cfg.fieldPtr(f).(type) {
	case *string:
		if len(f.choices) > 0 {
			// Verify that value is one of the allowed choices.
			for _, choice := range f.choices {
				if choice == value {
					*ptr = value
					return nil
				}
			}
			return fmt.Errorf("invalid %q value %q", f.name, value)
		}
		*ptr = value
	case *int:
		v, err := strconv.Atoi(value)
		if err != nil {
			return err
		}
		*ptr = v
	case *float64:
		v, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return err
		}
		*ptr = v
	case *bool:
		v, err := stringToBool(value)
		if err != nil {
			return err
		}
		*ptr = v
	default:
		panic(fmt.Sprintf("unsupported config field type %v", f.field.Type))
	}
	return nil
}

// isConfigurable returns true if name is either the name of a config field, or
// a valid value for a multi-choice config field.
func isConfigurable(name string) bool {
	_, ok := configFieldMap[name]
	return ok
}

// isBoolConfig returns true if name is either name of a boolean config field,
// or a valid value for a multi-choice config field.
func isBoolConfig(name string) bool {
	f, ok := configFieldMap[name]
	if !ok {
		return false
	}
	if name != f.name {
		return true // name must be one possible value for the field
	}
	var cfg config
	_, ok = cfg.fieldPtr(f).(*bool)
	return ok
}

// completeConfig returns the list of configurable names starting with prefix.
func completeConfig(prefix string) []string {
	var result []string
	for v := range configFieldMap {
		if strings.HasPrefix(v, prefix) {
			result = append(result, v)
		}
	}
	return result
}

// configure stores the name=value mapping into the current config, correctly
// handling the case when name identifies a particular choice in a field.
func configure(name, value string) error {
	currentMu.Lock()
	defer currentMu.Unlock()
	f, ok := configFieldMap[name]
	if !ok {
		return fmt.Errorf("unknown config field %q", name)
	}
	if f.name == name {
		return currentCfg.set(f, value)
	}
	// name must be one of the choices. If value is true, set field-value
	// to name.
	if v, err := strconv.ParseBool(value); v && err == nil {
		return currentCfg.set(f, name)
	}
	return fmt.Errorf("unknown config field %q", name)
}

// resetTransient sets all transient fields in *cfg to their currently
// configured values.
func (cfg *config) resetTransient() {
	current := currentConfig()
	cfg.Output = current.Output
	cfg.SourcePath = current.SourcePath
	cfg.TrimPath = current.TrimPath
	cfg.DivideBy = current.DivideBy
	cfg.SampleIndex = current.SampleIndex
}

// applyURL updates *cfg based on params.
func (cfg *config) applyURL(params url.Values) error {
	for _, f := range configFields {
		var value string
		if f.urlparam != "" {
			value = params.Get(f.urlparam)
		}
		if value == "" {
			continue
		}
		if err := cfg.set(f, value); err != nil {
			return fmt.Errorf("error setting config field %s: %v", f.name, err)
		}
	}
	return nil
}

// makeURL returns a URL based on initialURL that contains the config contents
// as parameters.  The second result is true iff a parameter value was changed.
func (cfg *config) makeURL(initialURL url.URL) (url.URL, bool) {
	q := initialURL.Query()
	changed := false
	for _, f := range configFields {
		if f.urlparam == "" || !f.saved {
			continue
		}
		v := cfg.get(f)
		if v == f.defaultValue {
			v = "" // URL for of default value is the empty string.
		} else if f.field.Type.Kind() == reflect.Bool {
			// Shorten bool values to "f" or "t"
			v = v[:1]
		}
		if q.Get(f.urlparam) == v {
			continue
		}
		changed = true
		if v == "" {
			q.Del(f.urlparam)
		} else {
			q.Set(f.urlparam, v)
		}
	}
	if changed {
		initialURL.RawQuery = q.Encode()
	}
	return initialURL, changed
}
