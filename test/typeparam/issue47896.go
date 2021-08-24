//  compile -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"database/sql"
)

// Collection generic interface which things can be added to.
type Collection[T any] interface {
	Add(T)
}

// Slice generic slice implementation of a Collection
type Slice[T any] []*T

func (s *Slice[T]) Add(t *T) {
	*s = append(*s, t)
}

type Scanner interface {
	Scan(...interface{}) error
}

type Mapper[T any] func(s Scanner, t T) error

type Repository[T any] struct {
	db *sql.DB
}

func (r *Repository[T]) scan(rows *sql.Rows, m Mapper[*T], c Collection[*T]) error {
	for rows.Next() {
		t := new(T)
		if err := m(rows, t); err != nil {
			return err
		}
		c.Add(t)
	}
	return rows.Err()
}

func (r *Repository[T]) query(query string, m Mapper[*T], c Collection[*T]) error {
	rows, err := r.db.Query(query)
	if err != nil {
		return err
	}
	if err := r.scan(rows, m, c); err != nil {
		rows.Close()
		return err
	}
	return rows.Close()
}

type Actor struct {
	ActorID   uint16
	FirstName string
	LastName  string
}

type ActorRepository struct {
	r Repository[Actor]
}

func (ActorRepository) scan(s Scanner, a *Actor) error {
	return s.Scan(&a.ActorID, &a.FirstName, &a.LastName)
}

func (r *ActorRepository) SelectAll(c Collection[*Actor]) error {
	return r.r.query("SELECT `actor_id`, `first_name`, `last_name` FROM `actor` LIMIT 10", r.scan, c)
}
