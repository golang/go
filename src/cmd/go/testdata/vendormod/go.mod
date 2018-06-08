module m

replace x v1.0.0 => ./x

replace y v1.0.0 => ./y

replace z v1.0.0 => ./z

replace w v1.0.0 => ./w

require (
	w v1.0.0
	x v1.0.0
	y v1.0.0
	z v1.0.0
)
