// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package modpack

import (
	"archive/zip"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func ValidateArgument(arg []string) error {
	if len(arg) != 2 {
		return errors.New("wrong number of argument")
	}

	if err := Validate(arg[0], arg[1]); err != nil {
		return err
	}

	return nil
}

func Pack(args []string) {
	// TODO Check for errors
	packageInfo, _ := GeneratePackageInfo(args[0], args[1])

	tempDir := os.TempDir()
	absoluteTempFolder := tempDir + packageInfo.GetTempFolderName()

	createFolder(absoluteTempFolder)
	defer deleteTemporaryFolder(tempDir, packageInfo.GetTempFolderName())
	copyProject(".", absoluteTempFolder)
	compressFolder(tempDir+packageInfo.vcs, packageInfo.GetZipFileName())

	fmt.Println("Generated", packageInfo.GetZipFileName())
}

func createFolder(path string) {
	absoluteDestination := path
	err := os.MkdirAll(absoluteDestination, os.ModeDir|os.ModePerm)
	if err != nil {
		panic(err)
	}
}

func deleteTemporaryFolder(tempFolder string, destinationPath string) {
	fmt.Println("Cleaning up temporary folders")

	split := strings.Split(destinationPath, string(os.PathSeparator))

	err := os.RemoveAll(tempFolder + split[0])
	if err != nil {
		panic(err)
	}
}

func copyProject(sourcePath string, destinationPath string) {
	fmt.Println(destinationPath)
	filepath.Walk(sourcePath, func(path string, info os.FileInfo, err error) error {
		dst := destinationPath + string(os.PathSeparator) + path
		src := sourcePath + string(os.PathSeparator) + path
		if info.IsDir() {
			os.Mkdir(dst, os.ModeDir|os.ModePerm)
			return nil
		}
		copyFile(src, dst)
		return nil
	})
}

func copyFile(source string, destination string) {
	in, err := os.Open(source)
	if err != nil {
		return
	}
	defer in.Close()

	out, err := os.Create(destination)
	if err != nil {
		return
	}
	defer out.Close()

	_, err = io.Copy(out, in)
	if err != nil {
		return
	}
	return
}

func compressFolder(folderToZip string, name string) {
	newZipFile, err := os.Create(name)
	if err != nil {
		panic(err)
	}

	writer := zip.NewWriter(newZipFile)
	defer writer.Close()

	_ = filepath.Walk(folderToZip, func(path string, info os.FileInfo, err error) error {
		if !info.IsDir() && !isSelf(info, name) {
			addFileToZip(writer, path)
		}
		return nil
	})
}

func isSelf(info os.FileInfo, name string) bool {
	return info.Name() == name
}

func addFileToZip(writer *zip.Writer, file string) error {
	openedFile, err := os.Open(file)
	if err != nil {
		return fmt.Errorf("unable to open file %v: %v", file, err)
	}
	defer openedFile.Close()

	// Strip the temp folder names
	fileToCreate := strings.Replace(file, os.TempDir(), "", 1)
	wr, err := writer.Create(fileToCreate)
	if err != nil {
		return fmt.Errorf("error adding file; '%s' to zip : %s", file, err)
	}

	if _, err := io.Copy(wr, openedFile); err != nil {
		return fmt.Errorf("error writing %s to zip: %s", file, err)
	}
	return nil
}
