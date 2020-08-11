package fsys

import (
	"cmd/go/internal/txtar"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

// initOverlay resets the overlay state to reflect the config.
// config should be a text archive string. The comment is the overlay config
// json, and the files, in the archive are laid out in a temp directory
// that cwd is set to.
func initOverlay(t *testing.T, config string) {
	t.Helper()

	// Create a temporary directory and chdir to it.
	prevwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	cwd = t.TempDir()
	if err := os.Chdir(cwd); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		overlay = nil
		if err := os.Chdir(prevwd); err != nil {
			t.Fatal(err)
		}
	})

	a := txtar.Parse([]byte(config))
	for _, f := range a.Files {
		name := filepath.Join(cwd, f.Name)
		if err := os.MkdirAll(filepath.Dir(name), 0777); err != nil {
			t.Fatal(err)
		}
		if err := ioutil.WriteFile(name, f.Data, 0666); err != nil {
			t.Fatal(err)
		}
	}

	var overlayJSON OverlayJSON
	if err := json.Unmarshal(a.Comment, &overlayJSON); err != nil {
		t.Fatal(fmt.Errorf("parsing overlay JSON: %v", err))
	}

	initFromJSON(overlayJSON)
}

func TestIsDir(t *testing.T) {
	initOverlay(t, `
{
	"Replace": {
		"subdir2/file2.txt":  "overlayfiles/subdir2_file2.txt",
		"subdir4":            "overlayfiles/subdir4",
		"subdir3/file3b.txt": "overlayfiles/subdir3_file3b.txt",
		"subdir5":            "",
		"subdir6":            ""
	}
}
-- subdir1/file1.txt --

-- subdir3/file3a.txt --
33
-- subdir4/file4.txt --
444
-- overlayfiles/subdir2_file2.txt --
2
-- overlayfiles/subdir3_file3b.txt --
66666
-- overlayfiles/subdir4 --
x
-- subdir6/file6.txt --
six
`)

	testCases := []struct {
		path          string
		want, wantErr bool
	}{
		{"", true, true},
		{".", true, false},
		{cwd, true, false},
		{cwd + string(filepath.Separator), true, false},
		// subdir1 is only on disk
		{filepath.Join(cwd, "subdir1"), true, false},
		{"subdir1", true, false},
		{"subdir1" + string(filepath.Separator), true, false},
		{"subdir1/file1.txt", false, false},
		{"subdir1/doesntexist.txt", false, true},
		{"doesntexist", false, true},
		// subdir2 is only in overlay
		{filepath.Join(cwd, "subdir2"), true, false},
		{"subdir2", true, false},
		{"subdir2" + string(filepath.Separator), true, false},
		{"subdir2/file2.txt", false, false},
		{"subdir2/doesntexist.txt", false, true},
		// subdir3 has files on disk and in overlay
		{filepath.Join(cwd, "subdir3"), true, false},
		{"subdir3", true, false},
		{"subdir3" + string(filepath.Separator), true, false},
		{"subdir3/file3a.txt", false, false},
		{"subdir3/file3b.txt", false, false},
		{"subdir3/doesntexist.txt", false, true},
		// subdir4 is overlaid with a file
		{filepath.Join(cwd, "subdir4"), false, false},
		{"subdir4", false, false},
		{"subdir4" + string(filepath.Separator), false, false},
		{"subdir4/file4.txt", false, false},
		{"subdir4/doesntexist.txt", false, false},
		// subdir5 doesn't exist, and is overlaid with a "delete" entry
		{filepath.Join(cwd, "subdir5"), false, false},
		{"subdir5", false, false},
		{"subdir5" + string(filepath.Separator), false, false},
		{"subdir5/file5.txt", false, false},
		{"subdir5/doesntexist.txt", false, false},
		// subdir6 does exist, and is overlaid with a "delete" entry
		{filepath.Join(cwd, "subdir6"), false, false},
		{"subdir6", false, false},
		{"subdir6" + string(filepath.Separator), false, false},
		{"subdir6/file6.txt", false, false},
		{"subdir6/doesntexist.txt", false, false},
	}

	for _, tc := range testCases {
		got, err := IsDir(tc.path)
		if err != nil {
			if !tc.wantErr {
				t.Errorf("IsDir(%q): got error with string %q, want no error", tc.path, err.Error())
			}
			continue
		}
		if tc.wantErr {
			t.Errorf("IsDir(%q): got no error, want error", tc.path)
		}
		if tc.want != got {
			t.Errorf("IsDir(%q) = %v, want %v", tc.path, got, tc.want)
		}
	}
}

func TestReadDir(t *testing.T) {
	initOverlay(t, `
{
	"Replace": {
		"subdir2/file2.txt":                 "overlayfiles/subdir2_file2.txt",
		"subdir4":                           "overlayfiles/subdir4",
		"subdir3/file3b.txt":                "overlayfiles/subdir3_file3b.txt",
		"subdir5":                           "",
		"subdir6/asubsubdir/afile.txt":      "overlayfiles/subdir6_asubsubdir_afile.txt",
		"subdir6/asubsubdir/zfile.txt":      "overlayfiles/subdir6_asubsubdir_zfile.txt",
		"subdir6/zsubsubdir/file.txt":       "overlayfiles/subdir6_zsubsubdir_file.txt",
		"subdir7/asubsubdir/file.txt":       "overlayfiles/subdir7_asubsubdir_file.txt",
		"subdir7/zsubsubdir/file.txt":       "overlayfiles/subdir7_zsubsubdir_file.txt",
		"subdir8/doesntexist":               "this_file_doesnt_exist_anywhere",
		"other/pointstodir":                 "overlayfiles/this_is_a_directory",
		"parentoverwritten/subdir1":         "overlayfiles/parentoverwritten_subdir1",
		"subdir9/this_file_is_overlaid.txt": "overlayfiles/subdir9_this_file_is_overlaid.txt",
		"subdir10/only_deleted_file.txt":    "",
		"subdir11/deleted.txt":              "",
		"subdir11":                          "overlayfiles/subdir11",
		"textfile.txt/file.go":              "overlayfiles/textfile_txt_file.go"
	}
}
-- subdir1/file1.txt --

-- subdir3/file3a.txt --
33
-- subdir4/file4.txt --
444
-- subdir6/file.txt --
-- subdir6/asubsubdir/file.txt --
-- subdir6/anothersubsubdir/file.txt --
-- subdir9/this_file_is_overlaid.txt --
-- subdir10/only_deleted_file.txt --
this will be deleted in overlay
-- subdir11/deleted.txt --
-- parentoverwritten/subdir1/subdir2/subdir3/file.txt --
-- textfile.txt --
this will be overridden by textfile.txt/file.go
-- overlayfiles/subdir2_file2.txt --
2
-- overlayfiles/subdir3_file3b.txt --
66666
-- overlayfiles/subdir4 --
x
-- overlayfiles/subdir6_asubsubdir_afile.txt --
-- overlayfiles/subdir6_asubsubdir_zfile.txt --
-- overlayfiles/subdir6_zsubsubdir_file.txt --
-- overlayfiles/subdir7_asubsubdir_file.txt --
-- overlayfiles/subdir7_zsubsubdir_file.txt --
-- overlayfiles/parentoverwritten_subdir1 --
x
-- overlayfiles/subdir9_this_file_is_overlaid.txt --
99999999
-- overlayfiles/subdir11 --
-- overlayfiles/this_is_a_directory/file.txt --
-- overlayfiles/textfile_txt_file.go --
x
`)

	testCases := map[string][]struct {
		name  string
		size  int64
		isDir bool
	}{
		".": {
			{"other", 0, true},
			{"overlayfiles", 0, true},
			{"parentoverwritten", 0, true},
			{"subdir1", 0, true},
			{"subdir10", 0, true},
			{"subdir11", 0, false},
			{"subdir2", 0, true},
			{"subdir3", 0, true},
			{"subdir4", 2, false},
			// no subdir5.
			{"subdir6", 0, true},
			{"subdir7", 0, true},
			{"subdir8", 0, true},
			{"subdir9", 0, true},
			{"textfile.txt", 0, true},
		},
		"subdir1": {{"file1.txt", 1, false}},
		"subdir2": {{"file2.txt", 2, false}},
		"subdir3": {{"file3a.txt", 3, false}, {"file3b.txt", 6, false}},
		"subdir6": {
			{"anothersubsubdir", 0, true},
			{"asubsubdir", 0, true},
			{"file.txt", 0, false},
			{"zsubsubdir", 0, true},
		},
		"subdir6/asubsubdir": {{"afile.txt", 0, false}, {"file.txt", 0, false}, {"zfile.txt", 0, false}},
		"subdir8":            {{"doesntexist", 0, false}}, // entry is returned even if destination file doesn't exist
		// check that read dir actually redirects files that already exist
		// the original this_file_is_overlaid.txt is empty
		"subdir9":           {{"this_file_is_overlaid.txt", 9, false}},
		"subdir10":          {},
		"parentoverwritten": {{"subdir1", 2, false}},
		"textfile.txt":      {{"file.go", 2, false}},
	}

	for dir, want := range testCases {
		fis, err := ReadDir(dir)
		if err != nil {
			t.Fatalf("ReadDir(%q): got error %q, want no error", dir, err)
		}
		if len(fis) != len(want) {
			t.Fatalf("ReadDir(%q) result: got %v entries; want %v entries", dir, len(fis), len(want))
		}
		for i := range fis {
			if fis[i].Name() != want[i].name {
				t.Fatalf("ReadDir(%q) entry %v: got Name() = %v, want %v", dir, i, fis[i].Name(), want[i].name)
			}
			if fis[i].IsDir() != want[i].isDir {
				t.Fatalf("ReadDir(%q) entry %v: got IsDir() = %v, want %v", dir, i, fis[i].IsDir(), want[i].isDir)
			}
			if want[i].isDir {
				// We don't try to get size right for directories
				continue
			}
			if fis[i].Size() != want[i].size {
				t.Fatalf("ReadDir(%q) entry %v: got Size() = %v, want %v", dir, i, fis[i].Size(), want[i].size)
			}
		}
	}

	errCases := []string{
		"subdir1/file1.txt", // regular file on disk
		"subdir2/file2.txt", // regular file in overlay
		"subdir4",           // directory overlaid with regular file
		"subdir5",           // directory deleted in overlay
		"parentoverwritten/subdir1/subdir2/subdir3", // parentoverwritten/subdir1 overlaid with regular file
		"parentoverwritten/subdir1/subdir2",         // parentoverwritten/subdir1 overlaid with regular file
		"subdir11",                                  // directory with deleted child, overlaid with regular file
		"other/pointstodir",
	}

	for _, dir := range errCases {
		_, gotErr := ReadDir(dir)
		if gotErr == nil {
			t.Errorf("ReadDir(%q): got no error, want error", dir)
		} else if _, ok := gotErr.(*os.PathError); !ok {
			t.Errorf("ReadDir(%q): got error with string %q and type %T, want os.PathError", dir, gotErr.Error(), gotErr)
		}
	}
}

func TestOverlayPath(t *testing.T) {
	initOverlay(t, `
{
	"Replace": {
		"subdir2/file2.txt":                 "overlayfiles/subdir2_file2.txt",
		"subdir3/doesntexist":               "this_file_doesnt_exist_anywhere",
		"subdir4/this_file_is_overlaid.txt": "overlayfiles/subdir4_this_file_is_overlaid.txt",
		"subdir5/deleted.txt":               "",
		"parentoverwritten/subdir1":         ""
	}
}
-- subdir1/file1.txt --
file 1
-- subdir4/this_file_is_overlaid.txt --
these contents are replaced by the overlay
-- parentoverwritten/subdir1/subdir2/subdir3/file.txt --
-- subdir5/deleted.txt --
deleted
-- overlayfiles/subdir2_file2.txt --
file 2
-- overlayfiles/subdir4_this_file_is_overlaid.txt --
99999999
`)

	testCases := []struct {
		path     string
		wantPath string
		wantOK   bool
	}{
		{"subdir1/file1.txt", "subdir1/file1.txt", false},
		// OverlayPath returns false for directories
		{"subdir2", "subdir2", false},
		{"subdir2/file2.txt", filepath.Join(cwd, "overlayfiles/subdir2_file2.txt"), true},
		// OverlayPath doesn't stat a file to see if it exists, so it happily returns
		// the 'to' path and true even if the 'to' path doesn't exist on disk.
		{"subdir3/doesntexist", filepath.Join(cwd, "this_file_doesnt_exist_anywhere"), true},
		// Like the subdir2/file2.txt case above, but subdir4 exists on disk, but subdir2 does not.
		{"subdir4/this_file_is_overlaid.txt", filepath.Join(cwd, "overlayfiles/subdir4_this_file_is_overlaid.txt"), true},
		{"subdir5", "subdir5", false},
		{"subdir5/deleted.txt", "", true},
	}

	for _, tc := range testCases {
		gotPath, gotOK := OverlayPath(tc.path)
		if gotPath != tc.wantPath || gotOK != tc.wantOK {
			t.Errorf("OverlayPath(%q): got %v, %v; want %v, %v",
				tc.path, gotPath, gotOK, tc.wantPath, tc.wantOK)
		}
	}
}

func TestOpen(t *testing.T) {
	initOverlay(t, `
{
    "Replace": {
		"subdir2/file2.txt":                  "overlayfiles/subdir2_file2.txt",
		"subdir3/doesntexist":                "this_file_doesnt_exist_anywhere",
		"subdir4/this_file_is_overlaid.txt":  "overlayfiles/subdir4_this_file_is_overlaid.txt",
		"subdir5/deleted.txt":                "",
		"parentoverwritten/subdir1":          "",
		"childoverlay/subdir1.txt/child.txt": "overlayfiles/child.txt",
		"subdir11/deleted.txt":               "",
		"subdir11":                           "overlayfiles/subdir11",
		"parentdeleted":                      "",
		"parentdeleted/file.txt":             "overlayfiles/parentdeleted_file.txt"
	}
}
-- subdir11/deleted.txt --
-- subdir1/file1.txt --
file 1
-- subdir4/this_file_is_overlaid.txt --
these contents are replaced by the overlay
-- parentoverwritten/subdir1/subdir2/subdir3/file.txt --
-- childoverlay/subdir1.txt --
this file doesn't exist because the path
childoverlay/subdir1.txt/child.txt is in the overlay
-- subdir5/deleted.txt --
deleted
-- parentdeleted --
this will be deleted so that parentdeleted/file.txt can exist
-- overlayfiles/subdir2_file2.txt --
file 2
-- overlayfiles/subdir4_this_file_is_overlaid.txt --
99999999
-- overlayfiles/child.txt --
-- overlayfiles/subdir11 --
11
-- overlayfiles/parentdeleted_file.txt --
this can exist because the parent directory is deleted
`)

	testCases := []struct {
		path         string
		wantContents string
		isErr        bool
	}{
		{"subdir1/file1.txt", "file 1\n", false},
		{"subdir2/file2.txt", "file 2\n", false},
		{"subdir3/doesntexist", "", true},
		{"subdir4/this_file_is_overlaid.txt", "99999999\n", false},
		{"subdir5/deleted.txt", "", true},
		{"parentoverwritten/subdir1/subdir2/subdir3/file.txt", "", true},
		{"childoverlay/subdir1.txt", "", true},
		{"subdir11", "11\n", false},
		{"parentdeleted/file.txt", "this can exist because the parent directory is deleted\n", false},
	}

	for _, tc := range testCases {
		f, err := Open(tc.path)
		if tc.isErr {
			if err == nil {
				f.Close()
				t.Errorf("Open(%q): got no error, but want error", tc.path)
			}
			continue
		}
		if err != nil {
			t.Errorf("Open(%q): got error %v, want nil", tc.path, err)
			continue
		}
		contents, err := ioutil.ReadAll(f)
		if err != nil {
			t.Errorf("unexpected error reading contents of file: %v", err)
		}
		if string(contents) != tc.wantContents {
			t.Errorf("contents of file opened with Open(%q): got %q, want %q",
				tc.path, contents, tc.wantContents)
		}
		f.Close()
	}
}

func TestIsDirWithGoFiles(t *testing.T) {
	initOverlay(t, `
{
	"Replace": {
		"goinoverlay/file.go":       "dummy",
		"directory/removed/by/file": "dummy",
		"directory_with_go_dir/dir.go/file.txt": "dummy",
		"otherdirectory/deleted.go": "",
		"nonexistentdirectory/deleted.go": "",
		"textfile.txt/file.go": "dummy"
	}
}
-- dummy --
a destination file for the overlay entries to point to
contents don't matter for this test
-- nogo/file.txt --
-- goondisk/file.go --
-- goinoverlay/file.txt --
-- directory/removed/by/file/in/overlay/file.go --
-- otherdirectory/deleted.go --
-- textfile.txt --
`)

	testCases := []struct {
		dir     string
		want    bool
		wantErr bool
	}{
		{"nogo", false, false},
		{"goondisk", true, false},
		{"goinoverlay", true, false},
		{"directory/removed/by/file/in/overlay", false, false},
		{"directory_with_go_dir", false, false},
		{"otherdirectory", false, false},
		{"nonexistentdirectory", false, false},
		{"textfile.txt", true, false},
	}

	for _, tc := range testCases {
		got, gotErr := IsDirWithGoFiles(tc.dir)
		if tc.wantErr {
			if gotErr == nil {
				t.Errorf("IsDirWithGoFiles(%q): got %v, %v; want non-nil error", tc.dir, got, gotErr)
			}
			continue
		}
		if gotErr != nil {
			t.Errorf("IsDirWithGoFiles(%q): got %v, %v; want nil error", tc.dir, got, gotErr)
		}
		if got != tc.want {
			t.Errorf("IsDirWithGoFiles(%q) = %v; want %v", tc.dir, got, tc.want)
		}
	}
}
