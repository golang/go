package user

import (
	"reflect"
	"testing"
)

var wantUsers = []*User{
	{
		Uid:      "-1",
		Gid:      "-1",
		Username: "adm",
		Name:     "adm",
		HomeDir:  "/usr/adm",
	},
	{
		Uid:      "0",
		Gid:      "0",
		Username: "none",
		Name:     "none",
		HomeDir:  "/usr/none",
	},
	{
		Uid:      "1",
		Gid:      "1",
		Username: "tor",
		Name:     "tor",
		HomeDir:  "/usr/tor",
	},
	{
		Uid:      "2",
		Gid:      "2",
		Username: "glenda",
		Name:     "glenda",
		HomeDir:  "/usr/glenda",
	},
	{
		Uid:      "9999",
		Gid:      "9999",
		Username: "noworld",
		Name:     "noworld",
		HomeDir:  "/usr/noworld",
	},
	{
		Uid:      "10000",
		Gid:      "10000",
		Username: "sys",
		Name:     "sys",
		HomeDir:  "/usr/sys",
	},
	{
		Uid:      "10001",
		Gid:      "10001",
		Username: "upas",
		Name:     "upas",
		HomeDir:  "/usr/upas",
	},
	{
		Uid:      "10002",
		Gid:      "10002",
		Username: "bootes",
		Name:     "bootes",
		HomeDir:  "/usr/bootes",
	},
	{
		Uid:      "test",
		Gid:      "test",
		Username: "test",
		Name:     "test",
		HomeDir:  "/usr/test",
	},
}

var wantGroups = []*Group{
	{Name: "adm", Gid: "-1"},
	{Name: "none", Gid: "0"},
	{Name: "tor", Gid: "1"},
	{Name: "glenda", Gid: "2"},
	{Name: "noworld", Gid: "9999"},
	{Name: "sys", Gid: "10000"},
	{Name: "upas", Gid: "10001"},
	{Name: "bootes", Gid: "10002"},
	{Name: "test", Gid: "test"},
}

const testUsersFile = "./testdata/plan9/users.txt"

func TestIterateUsers(t *testing.T) {
	saveTestUsersFile := usersFile
	defer func() {
		usersFile = saveTestUsersFile
	}()

	usersFile = testUsersFile

	gotUsers := make([]*User, 0, len(wantUsers))

	err := iterateUsers(func(user *User) error {
		gotUsers = append(gotUsers, user)
		return nil
	})

	if err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(wantUsers, gotUsers) {
		t.Errorf("iterate users result is incorrect: got: %+v, want: %+v", gotUsers, wantUsers)
	}
}

func TestIterateGroups(t *testing.T) {
	saveTestUsersFile := usersFile
	defer func() {
		usersFile = saveTestUsersFile
	}()

	usersFile = testUsersFile

	gotGroups := make([]*Group, 0, len(wantGroups))

	err := iterateGroups(func(groups *Group) error {
		gotGroups = append(gotGroups, groups)
		return nil
	})

	if err != nil {
		t.Error(err)
	}

	if !reflect.DeepEqual(wantGroups, gotGroups) {
		t.Errorf("iterate groups result is incorrect: got: %+v, want: %+v", gotGroups, wantGroups)
	}
}
