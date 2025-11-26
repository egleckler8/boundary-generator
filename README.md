Here is a crude README.md describing the algorithm(s) used to generate and 
vectorize a discrete polygonal boundary. It is scrapped together from big block 
comments I had in my code that had to go once I modularized the script into 
various functions in the `Boundary` class.

The `Boundary` class should now be very usable and hopefully flexible enough 
for our purposes. Ideally I will refine this file to include details and examples
of how to use `Boundary` for various things in various ways, but that is TODO for now

### (0) Boundary function

A map R^2 --> [-1, 1]

Do inquire if curious. I have spent long hours delving deeply into the dark arts
of "perlin noise." It's just smooth noise, and the boundary samples from it.
Hence, a smoothly random boundary function. Should be interesting for testing?



### (1) Place markers on the corners

This part is pretty chill. Rotate by a random amount and put a randomly long
vector down. Then rotate again and put down another random vector, etc.
Random ranges for theta and chosen artfully above so that the boundary is in
a goldilocks zone: not too erratic, and not too regular



### (2) Interpolate between corners

The way we do this is simply by making a line y = mx + b between each corner
and then for each x between the points, marking the appropriate y = mx + b.

BUT! This would leave gaps in the boundary. For example, imagine a line y = 0.2x.
How many times does this hit an integer over the interval [0, 5]? Only at (0,0),
and (5, 1). But, for the boundary to be closed, we would need at least a full 5
points marked on the grid.

This can be solved by also making a line x = my + b, and for each y between the
points, marking the appropriate x = my + b on the grid. The way the rounding
works sorts this out. The "if" statements make sure the boundary is closed.

ESSENTIALLY, THOUGH--we check the boundary index set so we never overwrite
points that are already in the boundary! This will totally screw up the
evaluation of the boundary function because the points from "y = mx + b"
will end up with a different boundary values than "x = ym + b" because of
weird rounding stuff. Thankfully checking set membership is O(1).



### (3) Make the boundary + interior

This is the part that really matters for FDM... We need some kind of output.,
and it will be the vectorized boundary and interior, ready for an FDM solve.

It was way harder than I thought to round up all the interior points! The
algorithm I came up with is a "wall insulation"/"quarantined plague" strategy.

(1) We know that the origin is within the boundary, so we put it in a queue.
(2) Then, until the queue is empty, we take an item out and check if it is a boundary
point. If not, and we haven't already looked at this point, then add it to the
interior points set.
(3) Then we put its four grid neighbors in queue to be checked; return to (2)

Hence, "plague strategy;" each point gets "infected." Thank heavens our boundary
is closed! It quarantines the exterior, and we end up with all the interior points.

In addition, we build a minimal boundary set. It could happen that above we
generate a boundary that is "two points thick" at some points," an easy
example to imagine would be very spiky corners. Now, as we go through all
the interior points, we can finally choose the boundary to be only the points
in the generated boundary set that are reachable by moving from the interior

FDM doesn't care what ordering we choose for the interior and boundary points.