package driver

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
)

// settings holds pprof settings.
type settings struct {
	// Configs holds a list of named UI configurations.
	Configs []namedConfig `json:"configs"`
}

// namedConfig associates a name with a config.
type namedConfig struct {
	Name string `json:"name"`
	config
}

// settingsFileName returns the name of the file where settings should be saved.
func settingsFileName() (string, error) {
	// Return "pprof/settings.json" under os.UserConfigDir().
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "pprof", "settings.json"), nil
}

// readSettings reads settings from fname.
func readSettings(fname string) (*settings, error) {
	data, err := os.ReadFile(fname)
	if err != nil {
		if os.IsNotExist(err) {
			return &settings{}, nil
		}
		return nil, fmt.Errorf("could not read settings: %w", err)
	}
	settings := &settings{}
	if err := json.Unmarshal(data, settings); err != nil {
		return nil, fmt.Errorf("could not parse settings: %w", err)
	}
	for i := range settings.Configs {
		settings.Configs[i].resetTransient()
	}
	return settings, nil
}

// writeSettings saves settings to fname.
func writeSettings(fname string, settings *settings) error {
	data, err := json.MarshalIndent(settings, "", "  ")
	if err != nil {
		return fmt.Errorf("could not encode settings: %w", err)
	}

	// create the settings directory if it does not exist
	// XDG specifies permissions 0700 when creating settings dirs:
	// https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
	if err := os.MkdirAll(filepath.Dir(fname), 0700); err != nil {
		return fmt.Errorf("failed to create settings directory: %w", err)
	}

	if err := os.WriteFile(fname, data, 0644); err != nil {
		return fmt.Errorf("failed to write settings: %w", err)
	}
	return nil
}

// configMenuEntry holds information for a single config menu entry.
type configMenuEntry struct {
	Name       string
	URL        string
	Current    bool // Is this the currently selected config?
	UserConfig bool // Is this a user-provided config?
}

// configMenu returns a list of items to add to a menu in the web UI.
func configMenu(fname string, u url.URL) []configMenuEntry {
	// Start with system configs.
	configs := []namedConfig{{Name: "Default", config: defaultConfig()}}
	if settings, err := readSettings(fname); err == nil {
		// Add user configs.
		configs = append(configs, settings.Configs...)
	}

	// Convert to menu entries.
	result := make([]configMenuEntry, len(configs))
	lastMatch := -1
	for i, cfg := range configs {
		dst, changed := cfg.config.makeURL(u)
		if !changed {
			lastMatch = i
		}
		// Use a relative URL to work in presence of stripping/redirects in webui.go.
		rel := &url.URL{RawQuery: dst.RawQuery, ForceQuery: true}
		result[i] = configMenuEntry{
			Name:       cfg.Name,
			URL:        rel.String(),
			UserConfig: (i != 0),
		}
	}
	// Mark the last matching config as current
	if lastMatch >= 0 {
		result[lastMatch].Current = true
	}
	return result
}

// editSettings edits settings by applying fn to them.
func editSettings(fname string, fn func(s *settings) error) error {
	settings, err := readSettings(fname)
	if err != nil {
		return err
	}
	if err := fn(settings); err != nil {
		return err
	}
	return writeSettings(fname, settings)
}

// setConfig saves the config specified in request to fname.
func setConfig(fname string, request url.URL) error {
	q := request.Query()
	name := q.Get("config")
	if name == "" {
		return fmt.Errorf("invalid config name")
	}
	cfg := currentConfig()
	if err := cfg.applyURL(q); err != nil {
		return err
	}
	return editSettings(fname, func { s ->
		for i, c := range s.Configs {
			if c.Name == name {
				s.Configs[i].config = cfg
				return nil
			}
		}
		s.Configs = append(s.Configs, namedConfig{Name: name, config: cfg})
		return nil
	})
}

// removeConfig removes config from fname.
func removeConfig(fname, config string) error {
	return editSettings(fname, func { s ->
		for i, c := range s.Configs {
			if c.Name == config {
				s.Configs = append(s.Configs[:i], s.Configs[i+1:]...)
				return nil
			}
		}
		return fmt.Errorf("config %s not found", config)
	})
}
