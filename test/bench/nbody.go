/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    * Neither the name of "The Computer Language Benchmarks Game" nor the
    name of "The Computer Language Shootout Benchmarks" nor the names of
    its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/* The Computer Language Benchmarks Game
 * http://shootout.alioth.debian.org/
 *
 * contributed by The Go Authors.
 * based on C program by Christoph Bauer
 */

package main

import (
	"flag"
	"fmt"
	"math"
)

var n = flag.Int("n", 1000, "number of iterations")

type Body struct {
	x, y, z, vx, vy, vz, mass float64
}

const (
	solarMass   = 4 * math.Pi * math.Pi
	daysPerYear = 365.24
)

func (b *Body) offsetMomentum(px, py, pz float64) {
	b.vx = -px / solarMass
	b.vy = -py / solarMass
	b.vz = -pz / solarMass
}

type System []*Body

func NewSystem(body []Body) System {
	n := make(System, len(body))
	for i := 0; i < len(body); i++ {
		n[i] = new(Body) // copy to avoid overwriting the inputs
		*n[i] = body[i]
	}
	var px, py, pz float64
	for _, body := range n {
		px += body.vx * body.mass
		py += body.vy * body.mass
		pz += body.vz * body.mass
	}
	n[0].offsetMomentum(px, py, pz)
	return n
}

func (sys System) energy() float64 {
	var e float64
	for i, body := range sys {
		e += 0.5 * body.mass *
			(body.vx*body.vx + body.vy*body.vy + body.vz*body.vz)
		for j := i + 1; j < len(sys); j++ {
			body2 := sys[j]
			dx := body.x - body2.x
			dy := body.y - body2.y
			dz := body.z - body2.z
			distance := math.Sqrt(dx*dx + dy*dy + dz*dz)
			e -= (body.mass * body2.mass) / distance
		}
	}
	return e
}

func (sys System) advance(dt float64) {
	for i, body := range sys {
		for j := i + 1; j < len(sys); j++ {
			body2 := sys[j]
			dx := body.x - body2.x
			dy := body.y - body2.y
			dz := body.z - body2.z

			dSquared := dx*dx + dy*dy + dz*dz
			distance := math.Sqrt(dSquared)
			mag := dt / (dSquared * distance)

			body.vx -= dx * body2.mass * mag
			body.vy -= dy * body2.mass * mag
			body.vz -= dz * body2.mass * mag

			body2.vx += dx * body.mass * mag
			body2.vy += dy * body.mass * mag
			body2.vz += dz * body.mass * mag
		}
	}

	for _, body := range sys {
		body.x += dt * body.vx
		body.y += dt * body.vy
		body.z += dt * body.vz
	}
}

var (
	jupiter = Body{
		x: 4.84143144246472090e+00,
		y: -1.16032004402742839e+00,
		z: -1.03622044471123109e-01,
		vx: 1.66007664274403694e-03 * daysPerYear,
		vy: 7.69901118419740425e-03 * daysPerYear,
		vz: -6.90460016972063023e-05 * daysPerYear,
		mass: 9.54791938424326609e-04 * solarMass,
	}
	saturn = Body{
		x: 8.34336671824457987e+00,
		y: 4.12479856412430479e+00,
		z: -4.03523417114321381e-01,
		vx: -2.76742510726862411e-03 * daysPerYear,
		vy: 4.99852801234917238e-03 * daysPerYear,
		vz: 2.30417297573763929e-05 * daysPerYear,
		mass: 2.85885980666130812e-04 * solarMass,
	}
	uranus = Body{
		x: 1.28943695621391310e+01,
		y: -1.51111514016986312e+01,
		z: -2.23307578892655734e-01,
		vx: 2.96460137564761618e-03 * daysPerYear,
		vy: 2.37847173959480950e-03 * daysPerYear,
		vz: -2.96589568540237556e-05 * daysPerYear,
		mass: 4.36624404335156298e-05 * solarMass,
	}
	neptune = Body{
		x: 1.53796971148509165e+01,
		y: -2.59193146099879641e+01,
		z: 1.79258772950371181e-01,
		vx: 2.68067772490389322e-03 * daysPerYear,
		vy: 1.62824170038242295e-03 * daysPerYear,
		vz: -9.51592254519715870e-05 * daysPerYear,
		mass: 5.15138902046611451e-05 * solarMass,
	}
	sun = Body{
		mass: solarMass,
	}
)

func main() {
	flag.Parse()

	system := NewSystem([]Body{sun, jupiter, saturn, uranus, neptune})
	fmt.Printf("%.9f\n", system.energy())
	for i := 0; i < *n; i++ {
		system.advance(0.01)
	}
	fmt.Printf("%.9f\n", system.energy())
}
