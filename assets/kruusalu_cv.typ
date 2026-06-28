#import "@preview/modern-cv:0.10.0": *


#show: resume.with(
  author: (
    firstname: "Mikk",
    lastname: "Kruusalu",
    email: "kruusalu.mikk@gmail.com",
    homepage: "https://mikruu.ee",
    phone: "53030327",
    github: "mikk-kruusalu",
    //twitter: "typstapp",
    //scholar: "",
    //orcid: "0000-0000-0000-000X",
    //birth: "January 1, 1990",
    linkedin: "mikk-kruusalu",
    //address: "111 Example St. Example City, EX 11111",
    positions: (
      "Machine learning",
      "Physics",
      "Numerical modelling",
    ),
  ),
  date: datetime.today().display(),
  language: "en",
  profile-picture: image("cv_profile.jpg", fit: "cover"),
  colored-headers: true,
  paper-size: "a4",
)

#columns(2, [


  = About
  MSc in Applied Physics and Data Science, passionate about modelling physical systems.
  I have engineering experience in control loop design, machine learning, embedded firmware and electronics.

  = Projects

  #resume-entry(
    title: "Implementing " + link("https://proceedings.mlr.press/v139/schutt21a/schutt21a.pdf")[PaiNN],
    date: github-link("MrCogito/Deep_learning_2023"),
    description: "Course project in DTU",
  )
  #resume-item[
    State of the art graph neural network architecture for predicting physical properties of molecules being much faster than conventional methods.
  ]

  #resume-entry(
    title: "Robotex",
  )
  #resume-item[
    - Competed in line following, water rally, maze solving, folkrace. I have won also a few prizes.
    - Building robots involved 3D modelling, custom PCBs, also algorithmic thinking and programming in C.
  ]

  #resume-entry(
    title: "Deep learning",
    date: github-link("mikk-kruusalu/deep_learning_project"),
    description: "Course project",
  )
  #resume-item[
    Implementations of different architectures such as CNN, RNN, GAN, VAE, PINN for various model problems.
  ]

  #resume-entry(
    title: "Nerve signal modelling",
    description: "Bachelor thesis",
    date: github-link("mikk-kruusalu/heimburg_jackson"),
  )
  #resume-item[
    I used Jax for analysing nonlinear dispersive wave phenomena in nerve axons. Learned about the electrophysiology of nerve signals.
  ]

  #resume-entry(
    title: "Graph topology optimisation",
    description: "Course project",
    date: link("https://mikruu.ee/assets/power-grid-optimisation.pdf"),
  )
  #resume-item[
    I analysed what is the optimal topology for a small scale deployable power grid to be tolerant for node attacks while delivering the most power.
  ]


  = Skills

  #resume-skill-item("Languages", (strong("Estonian"), "English (C1)"))
  #resume-skill-item(
    "Programming",
    (
      [Python -- Jax, Pytorch, Scipy, Numpy, Scikit-learn, Sktime,
        LaTeX, Git, Julia, Rust, R, C++],
    ),
  )
  #resume-skill-item(
    "Programs",
    ("Autodesk Fusion 360", "Kicad", "OpenFOAM", "ElmerFEM", "Paraview"),
  )
  #colbreak()

  = Education

  #resume-entry(
    title: "MSc Applied Physics and Data Science",
    date: "Sep 2024 - June 2026",
    description: "TalTech, Cum laude",
  )
  #resume-item[
    - Mathematical Modelling, Machine learning, Deep Learning, Numerical methods, Scientific computing
  ]
  #resume-entry(
    title: "Erasmus exchange student",
    date: "Sep 2023 - Jan 2024",
    description: "Denmark University of Techonolgy",
  )
  #resume-item[
    - Dynamical Systems, R, Deep learning, State Space Models
  ]
  #resume-entry(
    title: "BSc Applied Physics",
    date: "Sep 2021 - June 2024",
    description: "TalTech, Cum laude",
  )
  #resume-item[
    - Thesis -- modelling nerve signal propagation in an axon
    - Mathematical modelling, Physics, Probability and Statistics
  ]

  = Experience

  #resume-entry(
    title: "Development Engineer",
    date: "Mar 2023 - Present",
    description: "CAFA Tech",
  )
  #resume-item[
    - Custom visual inertial odometry system for position hold of a multirotor in GPS-denied environments.
    - Drone tether system control loop firmware and electornics.
    - Energy Team Lead in developing a small-scale microgrid.
    - Machine learning model for classifying objects based on their trajectories.
    - Developed a Data Acquisition System based on Beckhoff PLC and Timescale database.
    - Rapid prototyping of heat engines.
    - Developed a battery pack control system PCB and firmware.
  ]

  #resume-entry(
    title: "Robot Technician",
    date: "June 2022 - Feb 2023",
    description: "Starship Technologies",
  )
  #resume-item[
    - Repaired mechanics and electronics including PCBs
    - Create tools to improve the robot's repair process
  ]

  #resume-entry(
    title: "Bike mechanic",
    date: "June 2021 - Sep 2021",
    description: "Hawaii Express",
  )

]) // end columns
