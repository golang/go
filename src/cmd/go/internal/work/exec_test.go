package work

import (
	"cmd/go/internal/load"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func mockedExecCommand(command string, args ...string) *exec.Cmd {
	commandArgs := []string{"-test.run=TestExecOutput","--",command}
	commandArgs = append(commandArgs, args...)
	cmd := exec.Command(os.Args[0], commandArgs...)

	expectedInput := ""
	desiredPkgConfigOutput := ""
	if nextExecuteIndex < len(expectedExecutes) {
		expectedInput = expectedExecutes[nextExecuteIndex].expectedInput
		desiredPkgConfigOutput = expectedExecutes[nextExecuteIndex].desiredPkgConfigOutput
	}
	nextExecuteIndex++

	cmd.Env = []string{
		"GO_WANT_HELPER_PROCESS=1",
		fmt.Sprintf("EXPECTED_INPUTS=%s", expectedInput),
		fmt.Sprintf("DESIRED_OUTPUT=%s", desiredPkgConfigOutput),
	}
	return cmd
}

type PkgConfigExecute struct {
	expectedInput string
	desiredPkgConfigOutput string
}

var expectedExecutes []*PkgConfigExecute
var nextExecuteIndex = 0

type GetPkgConfigTestCase struct {
	cgoConfig string
	expectedCflags []string
	expectedLdflags []string
	expectedErr error
	executes []*PkgConfigExecute
}

func TestGetPkgConfigFlags(t *testing.T) {
	//Mock exec.Command
	runOutExecCommand = mockedExecCommand
	defer func() { runOutExecCommand = exec.Command }()

	//Mock PKG_CONFIG envvar
	oldPkgConfig := os.Getenv("PKG_CONFIG")
	defer func() {
		err := os.Setenv("PKG_CONFIG",oldPkgConfig)
		if err != nil {
			t.Errorf("Could not return PKG_CONFIG envvar back to its original value: %v", err)
		}
	}()
	err := os.Setenv("PKG_CONFIG","")
	if err != nil {
		t.Errorf("Could not set PKG_CONFIG envvar to sane test value: %v", err)
	}

	//Define test cases
	testCases := []*GetPkgConfigTestCase{
		//Empty cgo config
		{cgoConfig: "", expectedCflags:[]string{}, expectedLdflags:[]string{}, expectedErr:nil, executes: []*PkgConfigExecute{}},
		//Simple single package
		{
			cgoConfig: "--define-variable=prefix=. SomePkg",
			expectedCflags:[]string{"-Isomedir"},
			expectedLdflags:[]string{"-Lsomedir"},
			expectedErr:nil,
			executes: []*PkgConfigExecute{
				&PkgConfigExecute{expectedInput:"pkg-config --cflags --define-variable=prefix=. -- SomePkg", desiredPkgConfigOutput:"-Isomedir"},
				&PkgConfigExecute{expectedInput:"pkg-config --libs --define-variable=prefix=. -- SomePkg",desiredPkgConfigOutput:"-Lsomedir"},
			},
		},
		//Single package with spaces in paths
		{
			cgoConfig: "single",
			expectedCflags:[]string{"-IC:/Program Files/Git/usr/local/include/single"},
			expectedLdflags:[]string{"-LC:/Program Files/Git/usr/local/lib","-lsingle"},
			expectedErr:nil,
			executes:[]*PkgConfigExecute{
				&PkgConfigExecute{expectedInput:"pkg-config --cflags -- single", desiredPkgConfigOutput:"-IC:/Program\\ Files/Git/usr/local/include/single"},
				&PkgConfigExecute{expectedInput:"pkg-config --libs -- single", desiredPkgConfigOutput:"-LC:/Program\\ Files/Git/usr/local/lib -lsingle"},
			},
		},
		//Two packages
		{
			cgoConfig:"single multi",
			expectedCflags:[]string{"-I/usr/include/single","-I/usr/include/multi"},
			expectedLdflags:[]string{"-L/usr/lib/single","-lsingle","-L/usr/lib/multi","-lmulti"},
			expectedErr:nil,
			executes:[]*PkgConfigExecute{
				&PkgConfigExecute{expectedInput:"pkg-config --cflags -- single multi", desiredPkgConfigOutput:"-I/usr/include/single\n-I/usr/include/multi"},
				&PkgConfigExecute{expectedInput:"pkg-config --libs -- single multi", desiredPkgConfigOutput:"-L/usr/lib/single -lsingle\n-L/usr/lib/multi -lmulti"},
			},
		},
		//With -- in #cgo
		{
			cgoConfig:"--define-variable=prefix=. -- package",
			expectedCflags:[]string{"-I/usr/include/package"},
			expectedLdflags:[]string{"-L/usr/lib/package","-lpackage"},
			expectedErr:nil,
			executes:[]*PkgConfigExecute{
				&PkgConfigExecute{expectedInput:"pkg-config --cflags --define-variable=prefix=. -- package", desiredPkgConfigOutput:"-I/usr/include/package"},
				&PkgConfigExecute{expectedInput:"pkg-config --libs --define-variable=prefix=. -- package", desiredPkgConfigOutput:"-L/usr/lib/package -lpackage"},
			},
		},
		//Invalid package name in #cgo
		{
			cgoConfig:"@package",
			expectedErr:errors.New("invalid pkg-config package name: @package"),
		},
		//Insecure cflags output
		{
			cgoConfig:"badcflags",
			expectedErr:errors.New("invalid flag in pkg-config --cflags: -totallywrong"),
			executes:[]*PkgConfigExecute{
				&PkgConfigExecute{expectedInput:"pkg-config --cflags -- badcflags",desiredPkgConfigOutput:"-totallywrong"},
			},
		},
		//Insecure libs output
		{
			cgoConfig:"badldflags",
			expectedErr:errors.New("invalid flag in pkg-config --libs: -totallywrong"),
			executes:[]*PkgConfigExecute{
				&PkgConfigExecute{expectedInput:"pkg-config --cflags -- badldflags",desiredPkgConfigOutput:"-I/usr/lib/badldflags"},
				&PkgConfigExecute{expectedInput:"pkg-config --libs -- badldflags",desiredPkgConfigOutput:"-totallywrong"},
			},
		},
	}

	//Execute test cases
	for caseIndex, testCase := range testCases {
		expectedExecutes = testCase.executes
		nextExecuteIndex = 0
		builder := &Builder{}
		builder.Init()

		cflags, ldFlags, err := builder.getPkgConfigFlags(&load.Package{
			PackagePublic: load.PackagePublic{
				CgoPkgConfig: strings.Fields(testCase.cgoConfig),
			},
		})

		if nextExecuteIndex != len(expectedExecutes) {
			t.Errorf("Test Case %d: Expected exec.Command to be called %d times, but was called %d times",caseIndex, len(expectedExecutes), nextExecuteIndex)
		}

		if err != nil && testCase.expectedErr == nil {
			t.Errorf("Test Case %d: Expected success but got error: %s",caseIndex,err.Error())
		} else if err == nil && testCase.expectedErr != nil {
			t.Errorf("Test Case %d: Got success, but expected error: %s",caseIndex, testCase.expectedErr.Error())
		} else if err != nil && testCase.expectedErr != nil && err.Error() != testCase.expectedErr.Error() {
			t.Errorf("Test Case %d: Expected error: %s\nGot error %s",caseIndex, testCase.expectedErr.Error(), err.Error())
		} else if err == nil {
			cflagsMatch := true
			ldflagsMatch := true

			//CFLAGS and LDFLAGS need to contain all the same values, but not necessarily in any particular order
			if len(testCase.expectedCflags) != len(cflags) {
				cflagsMatch = false
			}
			if len(testCase.expectedLdflags) != len(ldFlags) {
				ldflagsMatch = false
			}

			if cflagsMatch {
				//Set compare cflags
				cflagsMap := make(map[string]bool)
				for _, cflag := range cflags {
					cflagsMap[cflag] = true
				}

				for _, flag := range testCase.expectedCflags {
					if !cflagsMap[flag] {
						cflagsMatch = false
						break
					}
				}
			}

			if ldflagsMatch {
				//Set compare ldflags
				ldflagsMap := make(map[string]bool)
				for _, ldflag := range ldFlags {
					ldflagsMap[ldflag] = true
				}

				for _, flag := range testCase.expectedLdflags {
					if !ldflagsMap[flag] {
						ldflagsMatch = false
						break
					}
				}
			}

			if !cflagsMatch {
				t.Errorf("Test Case %d: Expected CFLAGS: %s but got CFLAGS: %s",caseIndex,strings.Join(testCase.expectedCflags, " "), strings.Join(cflags, " "))
			}

			if !ldflagsMatch {
				t.Errorf("Test Case %d: Expected LDFLAGS: %s but got LDFLAGS: %s",caseIndex,strings.Join(testCase.expectedLdflags, " "), strings.Join(ldFlags, " "))
			}
		}
	}
}

func TestExecOutput(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}

	defer func() {
		//Suppress go test output to make this seem like a real pkg-config executable
		//This doesn't break tests later in the battery because this thing is only called from exec.Command
		//It's pretty nutty but it was the only way to make exec.go code testable in a self-contained way
		//This is only here to stop PASS from being output back to the calling code
		null, _ := os.Open(os.DevNull)
		os.Stdout = null
		os.Stderr = null
	}()

	expectedArgs := strings.Fields(os.Getenv("EXPECTED_INPUTS"))
	if len(expectedArgs) == 0 {
		t.Error("Attempted to test an exec.Command call with no EXPECTED_INPUTS value")
		return
	}

	//The command that the exec.go code attempted to execute comes after the first --
	//Check that it matches the command passed in with EXPECTED_INPUTS
	startedArgs := false
	nextExpectedArgs := 0
	for _, osArg := range os.Args {
		if !startedArgs {
			if osArg == "--" {
				startedArgs = true
			}
			continue
		}

		if nextExpectedArgs >= len(expectedArgs) || expectedArgs[nextExpectedArgs] != osArg {
			t.Errorf("Unexpected Input: %s",osArg)
		} else {
			nextExpectedArgs++
		}
	}

	if nextExpectedArgs != len(expectedArgs) {
		t.Errorf("Fewer args than expected in %s call", expectedArgs[0])
	}

	_, err := fmt.Fprintln(os.Stdout, os.Getenv("DESIRED_OUTPUT"))
	if err != nil {
		t.Errorf("Failed to write desired output to stdout: %v",err)
	}
}
