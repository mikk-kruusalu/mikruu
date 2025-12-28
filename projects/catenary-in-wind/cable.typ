#import "@preview/cetz:0.4.2": canvas, draw, angle

// Set font to match your website's Source Sans or Fraunces
#set page(width: auto, height: auto, margin: 5pt)
#let dd = $upright(d)$

#canvas({
  import draw: *
  import angle: *

  // Scale the whole drawing
  scale(2)

  // Coordinates
  let A = (0, 2)
  let B = (0, 0)
  let C = (0.8, 0.6)
  
  // 1. Cable curve
  bezier(A, B, C, stroke: (thickness: 1pt, paint: black))

  // 2. Axes
  // x-axis (vertical in your TikZ)
  line((0, -0.1), (0, 2.5), mark: (end: "straight"), stroke: black, name: "x_axis")
  content((rel: (0.2, 0), to: "x_axis.end"), $x$)
  
  // y-axis (horizontal in your TikZ)
  line((0.1, 0), (-0.5, 0), mark: (end: "straight"), stroke: black, name: "y_axis")
  content((rel: (0, 0.2), to: "y_axis.end"), $y$)
  
  // x-bar tick
  line((0.05, 0.75), (-0.05, 0.75), stroke: black)
  content((-0.15, 0.75), $overline(x)$)

  // 3. Label points
  circle(A, radius: 0.02, fill: black)
  content((rel: (0.4, 0.1), to: A), $(x_0, y_0)$)
  
  circle(B, radius: 0.02, fill: black)
  content((rel: (0.3, -0.1), to: B), $(0, 0)$)

  // 4. Forces (Static Labels)
  line((0.6, 0.5), (0.6, 0), mark: (end: "straight"), stroke: blue + 1pt)
  content((0.75, 0.1), text(blue)[$F_g$])
  
  line((-0.6, 1), (-0.1, 1), mark: (end: "straight"), stroke: blue + 1pt)
  content((-0.5, 1.15), text(blue)[$F_w$])

  // 5. Small segment analysis (offset to the right)
  let D = (1.5, 0.2)
  let E = (2.0, 1.8)
  let F = (1.85, 0.8)
  
  bezier(D, E, F, stroke: (thickness: 1pt, paint: black))
  content((rel: (-0.3, -0.5), to: E), $dd s$)
  
  // Dashed components
  line(D, (rel: (0.5, 0)), stroke: (paint: gray, dash: "dashed"), name: "dy")
  content("dy.centroid", anchor: "north", $dd y$)
  
  line(E, (rel: (0, -1.6)), stroke: (paint: gray, dash: "dashed"), name: "dx")
  content("dx.centroid", anchor: "south-west", $dd x$)

  // Force Vectors on Segment
  line(D, (rel: (-0.2, -0.3)), mark: (end: "straight"), stroke: blue + 1pt)
  content((rel: (-0.2, -0.4), to: D), text(blue, size: 8pt)[$T(x,y)$])
  
  line(E, (rel: (0.04, 0.3)), mark: (end: "straight"), stroke: blue + 1pt)
  content((rel: (0, 0.4), to: E), text(blue, size: 8pt)[$T(x+dd x, y+dd y)$])
  
  // Gravity/Wind on segment
  line(F, (rel: (0, -0.3), to: F), mark: (end: "straight"), stroke: blue)
  content((rel: (0.2, -0.15), to: F), text(blue, size: 8pt)[$rho g dd s$])
  
  line((rel: (-0.4, 0.1), to: F), (rel: (0.3, 0)), mark: (end: "straight"), stroke: blue)
  content((rel: (-0.4, 0.2), to: F), text(blue, size: 8pt)[$f_w dd x$])

  // Added: Angle alpha between x-axis (vertical up) and ds direction
  line(D, (rel: (0.0, 0.55), to: D), stroke: (paint: gray, dash: "dashed"))
  angle(D, (rel: (-0.05, 0.0), to: F), (rel: (0.0, 1.0), to: D), label: $alpha$, radius: 0.5, inner: true, label-radius: 80%, direction: "ccw")
})
