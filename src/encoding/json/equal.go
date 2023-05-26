package json

import (
	"encoding/json"
	"reflect"
)

// equalJSON Determine whether two JSON strings are equivalent (with the same key value, in different order)
func equalJSON(jsonStr1, jsonStr2 string) (bool, error) {
	var i interface{}
	var i2 interface{}
	err := json.Unmarshal([]byte(jsonStr1), &i)
	if err != nil {
		return false, err
	}
	err = json.Unmarshal([]byte(jsonStr2), &i2)
	if err != nil {
		return false, err
	}
	return reflect.DeepEqual(i, i2), nil
}

// Equal Determine whether two or more JSON strings are equivalent (with the same key value and different order)
func Equal(jsonStr1, jsonStr2 string, MoreJSON ...string) (bool, error) {
	if len(MoreJSON) == 0 {
		return equalJSON(jsonStr1, jsonStr2)
	}
	equal, err := equalJSON(jsonStr1, jsonStr2)
	if err != nil {
		return false, err
	}
	if !equal {
		return false, nil
	}
	for _, js := range MoreJSON {
		eq, err := equalJSON(jsonStr1, js)
		if err != nil {
			return false, err
		}
		if !eq {
			return false, nil
		}
	}
	return true, nil
}
