# Prandtl-Meyer Expansion Wave Simulation

This project simulates a **Prandtl-Meyer expansion fan** — an essential phenomenon in supersonic compressible flow — where a flow turns around a convex corner and expands isentropically, increasing its Mach number. The simulation is implemented in Python and uses the **space-marching method**, following the methodology outlined in *John D. Anderson's "Introduction to Computational Fluid Dynamics: The Basics with Applications."*

---

## Theoretical Background

In supersonic flow, if a fluid element encounters a convex corner, it cannot follow a smooth curved path like in subsonic flow due to the hyperbolic nature of the governing equations. Instead, the flow expands through an **expansion fan** composed of infinitesimal **Mach waves**, known as a **Prandtl-Meyer expansion**.

Key properties:
- The expansion is **isentropic** (no entropy change).
- The flow **accelerates** and **turns**, increasing its Mach number and decreasing pressure and temperature.
- Governed by **compatibility equations** along characteristic lines.

The governing 2D steady supersonic Euler equations reduce to a **hyperbolic system**, which can be solved using the **method of characteristics** or simplified using **space marching** in a transformed coordinate system.

---

## Numerical Method

### Grid Generation
- The **physical domain** (expansion region) is mapped to a **computational domain** using structured grid generation.
- The transformation simplifies the geometry and makes it easier to implement boundary conditions and space marching.

### Space Marching Technique
- Since the flow is **supersonic everywhere**, information propagates only downstream.
- The solver "marches" in the x-direction, solving for flow variables at each step using upstream information.
- Compatible with Anderson's description of solving **steady supersonic flow** problems without requiring iteration in time.

### Key Assumptions
- Inviscid, compressible, steady, 2D flow
- Ideal gas behavior
- No shocks (expansion only)

---


