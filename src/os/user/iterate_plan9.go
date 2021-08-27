package user

import (
	"bytes"
	"fmt"
	"os"
	"strings"
)

// Users and groups file location in plan9. Since, this value is mutated during
// testing, to a path to testdata, it is on purpose not a constant.
var usersFile = "/adm/users"

// userGroupIterator is a helper iterator function, which parses /adm/users
func userGroupIterator(lineFn lineFunc) (err error) {
	f, err := os.Open(usersFile)
	if err != nil {
		return fmt.Errorf("open users file: %w", err)
	}
	defer f.Close()
	_, err = readColonFile(f, lineFn, 3)
	return
}

// parsePlan9UserGroup matches valid /adm/users line and provides colon split
// string slice to returnFn which can parse either *User or *Group.
// On plan9 /adm/users lines are both users and groups.
//
// Plan9 /adm/user line structure looks like this:
// id:name:leader:members
// sys:sys::glenda,mj <-- user/group without a leader, with 2 members glenda and mj
// mj:mj:: <-- user/group without a leader, without members
//
// According to plan 9 users(6): ids are arbitrary text strings, typically the same as name.
// In older Plan 9 file servers, ids are small decimal numbers.
func parsePlan9UserGroup(returnFn func([]string) interface{}) lineFunc {
	return func(line []byte) (v interface{}, err error) {
		if bytes.Count(line, []byte{':'}) < 3 {
			return
		}
		// id:name:leader:members
		// id can be negative (start with a "-" symbol) in plan 9.
		parts := strings.SplitN(string(line), ":", 4)
		if len(parts) < 4 || parts[0] == "" ||
			parts[0][0] == '+' {
			return
		}

		return returnFn(parts), nil
	}
}

// userReturnFn builds *User struct from provided parts
func userReturnFn(parts []string) interface{} {
	return &User{
		Uid:      parts[0],
		Gid:      parts[0],
		Username: parts[1],
		Name:     parts[1],

		// There is no clear documentation which directory is set to homedir for user.
		// However, when a new user is created, when user logs in, $HOME environment
		// variable is set to /usr/<username> and this is also the login directory.
		HomeDir: "/usr/" + parts[1],
	}
}

// groupReturnFn builds *Group struct from provided parts
func groupReturnFn(parts []string) interface{} {
	return &Group{Name: parts[1], Gid: parts[0]}
}

func iterateUsers(fn NextUserFunc) error {
	return userGroupIterator(func(line []byte) (interface{}, error) {
		v, _ := parsePlan9UserGroup(userReturnFn)(line)
		if user, ok := v.(*User); ok {
			err := fn(user)
			if err != nil {
				return nil, err
			}
		}
		return nil, nil
	})
}

func iterateGroups(fn NextGroupFunc) error {
	return userGroupIterator(func(line []byte) (interface{}, error) {
		v, _ := parsePlan9UserGroup(groupReturnFn)(line)
		if group, ok := v.(*Group); ok {
			err := fn(group)
			if err != nil {
				return nil, err
			}
		}
		return nil, nil
	})
}
