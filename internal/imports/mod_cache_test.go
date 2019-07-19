package imports

import (
	"fmt"
	"testing"
)

func TestDirectoryPackageInfoReachedStatus(t *testing.T) {
	tests := []struct {
		info       directoryPackageInfo
		target     directoryPackageStatus
		wantStatus bool
		wantError  bool
	}{
		{
			info: directoryPackageInfo{
				status: directoryScanned,
				err:    nil,
			},
			target:     directoryScanned,
			wantStatus: true,
		},
		{
			info: directoryPackageInfo{
				status: directoryScanned,
				err:    fmt.Errorf("error getting to directory scanned"),
			},
			target:     directoryScanned,
			wantStatus: true,
			wantError:  true,
		},
		{
			info:       directoryPackageInfo{},
			target:     directoryScanned,
			wantStatus: false,
		},
	}

	for _, tt := range tests {
		gotStatus, gotErr := tt.info.reachedStatus(tt.target)
		if gotErr != nil {
			if !tt.wantError {
				t.Errorf("unexpected error: %s", gotErr)
			}
			continue
		}

		if tt.wantStatus != gotStatus {
			t.Errorf("reached status expected: %v, got: %v", tt.wantStatus, gotStatus)
		}
	}
}
