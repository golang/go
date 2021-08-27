package user_test

import (
	"errors"
	"fmt"
	"os/user"
)

func ExampleIterateUsers() {
	stopErr := errors.New("stop iterating")
	// Get first 20 users
	users := make([]*user.User, 0, 20)
	i := 0
	err := user.IterateUsers(func(user *user.User) error {
		users = append(users, user)
		i++

		// Once we return non-nil error - iteration process stops
		if i >= 20 {
			return stopErr
		}

		// As long as error is nil, IterateUsers will iterate over users database
		return nil
	})

	if err != stopErr && err != nil {
		fmt.Printf("error encountered while iterating users database: %v", err)
	}

	// Here users slice can be used to do something with collected users.
}

func ExampleIterateGroups() {
	stopErr := errors.New("stop iterating")
	// Get first 20 groups
	groups := make([]*user.Group, 0, 20)
	i := 0
	err := user.IterateGroups(func(group *user.Group) error {
		groups = append(groups, group)
		i++

		// Once we return non-nil error - iteration process stops
		if i >= 20 {
			return stopErr
		}

		// As long as error is nil, IterateGroups will iterate over groups database
		return nil
	})

	if err != stopErr && err != nil {
		fmt.Printf("error encountered while iterating groups database: %v", err)
	}

	// Here groups slice can be used to do something with collected groups.
}
