package list_test

import (
	"cmd/go/internal/cfg"
	"cmd/go/internal/modload"
	"context"
	"internal/testenv"
	"io"
	"os"
	"path/filepath"
	"testing"
)

var modulesForTest = []struct {
	name            string
	testDataFolder  string
	expectedModules int
}{
	{
		name:            "Empty",
		testDataFolder:  "empty",
		expectedModules: 1,
	},
	{
		name:            "Cmd",
		testDataFolder:  "cmd",
		expectedModules: 13,
	},
	{
		name:            "K8S",
		testDataFolder:  "strippedk8s",
		expectedModules: 477,
	},
}

func BenchmarkListModules(b *testing.B) {
	testenv.MustHaveExternalNetwork(b)
	testDataDir := b.TempDir()
	for _, m := range modulesForTest {
		// move the modules outside the golang cmd module
		moduleTempDir, err := copyModulesToTempDir(testDataDir, m.testDataFolder)
		if err != nil {
			b.Fatalf("Failed to copy testdata files: %v", err)
		}
		b.Run(m.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				gopath := b.TempDir()
				os.Setenv("GOPATH", gopath)
				cfg.BuildContext.GOPATH = gopath
				cfg.GOMODCACHE = filepath.Join(gopath, "pkg/mod")
				cfg.SumdbDir = filepath.Join(gopath, "pkg/sumdb")
				modload.Reset()
				modload.ForceUseModules = true
				modload.RootMode = modload.NeedRoot
				cfg.ModulesEnabled = true
				cfg.ModFile = filepath.Join(moduleTempDir, "go.mod")
				cfg.BuildMod = "readonly"
				cfg.BuildModExplicit = true
				cfg.BuildModReason = "to avoid vendoring error"
				modload.Init()
				modules, err := modload.ListModules(context.Background(), []string{"all"}, 0, "")
				if err != nil {
					b.Fatalf("ListModules() error = %v", err)
				}
				if m.expectedModules != len(modules) {
					b.Fatalf("expected %d modules, got %d", m.expectedModules, len(modules))
				}
			}
		})
	}
}

func copyModulesToTempDir(testDataDir string, folder string) (string, error) {
	moduleTempDir := filepath.Join(testDataDir, folder)
	err := os.Mkdir(moduleTempDir, 0755)
	if err != nil {
		return "", err
	}
	err = copyFileToDirectory(filepath.Join("testdata", folder, "go.mod"), moduleTempDir)
	if err != nil {
		return "", err
	}
	goSum := filepath.Join("testdata", folder, "go.sum")
	if _, err = os.Stat(goSum); err == nil {
		err = copyFileToDirectory(goSum, moduleTempDir)
		if err != nil {
			return "", err
		}
	}
	return moduleTempDir, nil
}

func copyFileToDirectory(src, destDir string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	_, filename := filepath.Split(src)
	destPath := filepath.Join(destDir, filename)

	destinationFile, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer destinationFile.Close()

	_, err = io.Copy(destinationFile, sourceFile)
	if err != nil {
		return err
	}

	return nil
}
