/**
 * Multi-major definitions and pre-college credit mappings.
 * Currently detailed catalog for CS&E; other majors have placeholder structures
 * to demonstrate the major-selection flow.
 */

export const MAJORS = [
  {
    id: "cse",
    name: "Computer Science & Engineering",
    department: "Computer Science & Engineering",
    college: "College of Engineering",
    totalCredits: 124,
    description: "Design and build software systems, from AI to operating systems.",
  },
  {
    id: "ce",
    name: "Computer Engineering",
    department: "Electrical & Biomedical Engineering",
    college: "College of Engineering",
    totalCredits: 128,
    description: "Hardware-software integration, embedded systems, and digital design.",
  },
  {
    id: "ee",
    name: "Electrical Engineering",
    department: "Electrical & Biomedical Engineering",
    college: "College of Engineering",
    totalCredits: 128,
    description: "Circuits, signal processing, power systems, and electronics.",
  },
  {
    id: "me",
    name: "Mechanical Engineering",
    department: "Mechanical Engineering",
    college: "College of Engineering",
    totalCredits: 130,
    description: "Thermodynamics, mechanics, robotics, and manufacturing.",
  },
  {
    id: "math",
    name: "Mathematics",
    department: "Mathematics & Statistics",
    college: "College of Science",
    totalCredits: 120,
    description: "Pure and applied mathematics, statistics, and data analysis.",
  },
  {
    id: "phys",
    name: "Physics",
    department: "Physics",
    college: "College of Science",
    totalCredits: 120,
    description: "Classical and modern physics, quantum mechanics, and astrophysics.",
  },
];

/**
 * AP Exam credit equivalencies at UNR.
 * score: minimum AP score needed
 * grants: array of { courseId, credits } that the AP credit satisfies
 */
export const AP_CREDITS = [
  {
    examName: "AP Computer Science A",
    scores: [
      { minScore: 3, grants: [{ courseId: "cs135", credits: 3 }] },
      { minScore: 4, grants: [{ courseId: "cs135", credits: 3 }] },
      { minScore: 5, grants: [{ courseId: "cs135", credits: 3 }, { courseId: "cs202", credits: 3 }] },
    ],
  },
  {
    examName: "AP Computer Science Principles",
    scores: [
      { minScore: 3, grants: [{ courseId: "cs135", credits: 3 }] },
    ],
  },
  {
    examName: "AP Calculus AB",
    scores: [
      { minScore: 3, grants: [{ courseId: "math181", credits: 4 }] },
    ],
  },
  {
    examName: "AP Calculus BC",
    scores: [
      { minScore: 3, grants: [{ courseId: "math181", credits: 4 }] },
      { minScore: 4, grants: [{ courseId: "math181", credits: 4 }, { courseId: "math182", credits: 4 }] },
      { minScore: 5, grants: [{ courseId: "math181", credits: 4 }, { courseId: "math182", credits: 4 }] },
    ],
  },
  {
    examName: "AP Statistics",
    scores: [
      { minScore: 3, grants: [{ courseId: "stat352", credits: 3 }] },
    ],
  },
  {
    examName: "AP Physics C: Mechanics",
    scores: [
      { minScore: 3, grants: [{ courseId: "phys180", credits: 3 }, { courseId: "phys180l", credits: 1 }] },
    ],
  },
  {
    examName: "AP Physics C: E&M",
    scores: [
      { minScore: 3, grants: [{ courseId: "phys181", credits: 3 }, { courseId: "phys181l", credits: 1 }] },
    ],
  },
  {
    examName: "AP Physics 1",
    scores: [
      { minScore: 4, grants: [{ courseId: "phys180", credits: 3 }, { courseId: "phys180l", credits: 1 }] },
    ],
  },
  {
    examName: "AP Physics 2",
    scores: [
      { minScore: 4, grants: [{ courseId: "phys181", credits: 3 }, { courseId: "phys181l", credits: 1 }] },
    ],
  },
  {
    examName: "AP English Language",
    scores: [
      { minScore: 3, grants: [{ courseId: "eng101", credits: 3 }] },
      { minScore: 4, grants: [{ courseId: "eng101", credits: 3 }, { courseId: "eng102", credits: 3 }] },
    ],
  },
  {
    examName: "AP English Literature",
    scores: [
      { minScore: 3, grants: [{ courseId: "eng101", credits: 3 }] },
      { minScore: 4, grants: [{ courseId: "eng101", credits: 3 }, { courseId: "eng102", credits: 3 }] },
    ],
  },
  {
    examName: "AP Psychology",
    scores: [
      { minScore: 3, grants: [{ courseId: "psy101", credits: 3 }] },
    ],
  },
  {
    examName: "AP Macroeconomics",
    scores: [
      { minScore: 3, grants: [{ courseId: "econ102", credits: 3 }] },
    ],
  },
  {
    examName: "AP U.S. Government",
    scores: [
      { minScore: 3, grants: [{ courseId: "pols101", credits: 3 }] },
    ],
  },
];

/**
 * ACT/SAT Math placement — determines which math course a student starts with.
 */
export const MATH_PLACEMENT = [
  { test: "ACT Math", minScore: 28, placesInto: "math181", skips: [], label: "Ready for Calculus I" },
  { test: "ACT Math", minScore: 22, placesInto: "math127", skips: [], label: "Pre-Calculus required first" },
  { test: "SAT Math", minScore: 640, placesInto: "math181", skips: [], label: "Ready for Calculus I" },
  { test: "SAT Math", minScore: 530, placesInto: "math127", skips: [], label: "Pre-Calculus required first" },
];

/**
 * Common dual enrollment / transfer course equivalencies.
 */
export const DUAL_ENROLLMENT_MAP = [
  { transferName: "Intro to Programming / CS 1", equivalentId: "cs135" },
  { transferName: "Object-Oriented Programming / CS 2", equivalentId: "cs202" },
  { transferName: "Calculus I", equivalentId: "math181" },
  { transferName: "Calculus II", equivalentId: "math182" },
  { transferName: "Calculus III / Multivariable", equivalentId: "math283" },
  { transferName: "Linear Algebra", equivalentId: "math330" },
  { transferName: "Statistics / Probability", equivalentId: "stat352" },
  { transferName: "Discrete Mathematics", equivalentId: "cs252" },
  { transferName: "Physics I (Mechanics)", equivalentId: "phys180" },
  { transferName: "Physics I Lab", equivalentId: "phys180l" },
  { transferName: "Physics II (E&M)", equivalentId: "phys181" },
  { transferName: "Physics II Lab", equivalentId: "phys181l" },
  { transferName: "English Composition I", equivalentId: "eng101" },
  { transferName: "English Composition II", equivalentId: "eng102" },
  { transferName: "General Psychology", equivalentId: "psy101" },
  { transferName: "Intro to Sociology", equivalentId: "soc101" },
  { transferName: "Microeconomics", equivalentId: "econ102" },
  { transferName: "U.S. Government / American Politics", equivalentId: "pols101" },
  { transferName: "Art Appreciation", equivalentId: "art100" },
  { transferName: "Music Appreciation", equivalentId: "mus100" },
  { transferName: "Intro to Theatre", equivalentId: "thtr100" },
];

/**
 * Given AP exams with scores, returns the set of course IDs granted.
 */
export function resolveAPCredits(apEntries) {
  const granted = new Set();
  for (const entry of apEntries) {
    const exam = AP_CREDITS.find((e) => e.examName === entry.examName);
    if (!exam) continue;
    // Find the best matching score tier (highest minScore that the student meets)
    const applicable = exam.scores
      .filter((s) => entry.score >= s.minScore)
      .sort((a, b) => b.minScore - a.minScore);
    if (applicable.length > 0) {
      applicable[0].grants.forEach((g) => granted.add(g.courseId));
    }
  }
  return granted;
}

/**
 * Given math test scores, returns placement info.
 */
export function resolveMathPlacement(testType, score) {
  const placements = MATH_PLACEMENT.filter(
    (p) => p.test === testType && score >= p.minScore
  ).sort((a, b) => b.minScore - a.minScore);
  return placements.length > 0 ? placements[0] : null;
}
