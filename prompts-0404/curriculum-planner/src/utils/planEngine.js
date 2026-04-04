/**
 * Plan Engine – handles prerequisite checking, semester validation,
 * elective filtering, and cascading plan updates.
 */

import { courses, getCourse, CATEGORY, SEMESTER, degreeRequirements } from "../data/catalog";

/**
 * Checks if all prerequisites for a course are satisfied by the
 * courses completed in previous semesters of the plan.
 */
export function arePrerequisitesMet(courseId, semesterIndex, plan) {
  const course = getCourse(courseId);
  if (!course) return { met: false, missing: [courseId] };

  const completedBefore = new Set();
  for (let i = 0; i < semesterIndex; i++) {
    plan[i].forEach((id) => completedBefore.add(id));
  }

  const missing = course.prerequisites.filter((pid) => !completedBefore.has(pid));
  return { met: missing.length === 0, missing };
}

/**
 * Checks if co-requisites are satisfied (taken in the same semester or before).
 */
export function areCorequisitesMet(courseId, semesterIndex, plan) {
  const course = getCourse(courseId);
  if (!course) return { met: false, missing: [courseId] };

  const availableUpTo = new Set();
  for (let i = 0; i <= semesterIndex; i++) {
    plan[i].forEach((id) => availableUpTo.add(id));
  }

  const missing = (course.corequisites || []).filter((cid) => !availableUpTo.has(cid));
  return { met: missing.length === 0, missing };
}

/**
 * Returns whether a course is offered in the given semester index.
 * Even index = Fall, Odd index = Spring.
 */
export function isOfferedInSemester(courseId, semesterIndex) {
  const course = getCourse(courseId);
  if (!course) return false;
  const term = semesterIndex % 2 === 0 ? SEMESTER.FALL : SEMESTER.SPRING;
  return course.semesters.includes(SEMESTER.BOTH) || course.semesters.includes(term);
}

/**
 * Validates an entire plan, returning issues per semester per course.
 */
export function validatePlan(plan) {
  const issues = plan.map(() => ({}));

  plan.forEach((semCourses, semIdx) => {
    semCourses.forEach((courseId) => {
      const courseIssues = [];

      const prereqCheck = arePrerequisitesMet(courseId, semIdx, plan);
      if (!prereqCheck.met) {
        const names = prereqCheck.missing.map((id) => getCourse(id)?.code || id);
        courseIssues.push(`Missing prerequisites: ${names.join(", ")}`);
      }

      const coreqCheck = areCorequisitesMet(courseId, semIdx, plan);
      if (!coreqCheck.met) {
        const names = coreqCheck.missing.map((id) => getCourse(id)?.code || id);
        courseIssues.push(`Missing co-requisites: ${names.join(", ")}`);
      }

      if (!isOfferedInSemester(courseId, semIdx)) {
        const term = semIdx % 2 === 0 ? "Fall" : "Spring";
        const course = getCourse(courseId);
        courseIssues.push(`${course?.code} is not offered in ${term}`);
      }

      if (courseIssues.length > 0) {
        issues[semIdx][courseId] = courseIssues;
      }
    });
  });

  return issues;
}

/**
 * Returns all technical elective courses whose prerequisites can be met
 * given the courses completed/planned through the specified semester.
 */
export function getAvailableElectives(semesterIndex, plan) {
  const completedUpTo = new Set();
  for (let i = 0; i < semesterIndex; i++) {
    plan[i].forEach((id) => completedUpTo.add(id));
  }

  const alreadyPlanned = new Set(plan.flat());

  const techReq = degreeRequirements.find((r) => r.name === "Technical Electives");
  if (!techReq) return [];

  return techReq.electivePool
    .map((id) => getCourse(id))
    .filter((c) => c && !alreadyPlanned.has(c.id))
    .filter((c) => {
      const prereqsMet = c.prerequisites.every((pid) => completedUpTo.has(pid));
      const offeredHere = isOfferedInSemester(c.id, semesterIndex);
      return prereqsMet && offeredHere;
    });
}

/**
 * Returns substitution options for a flexible course slot.
 */
export function getSubstitutions(courseId) {
  const course = getCourse(courseId);
  if (!course || !course.substitutions) return [];
  return course.substitutions.map((id) => getCourse(id)).filter(Boolean);
}

/**
 * Finds all courses in the plan that depend on a given course
 * (directly or transitively as a prerequisite).
 */
export function getDependents(courseId, plan) {
  const dependents = new Set();
  const queue = [courseId];

  while (queue.length > 0) {
    const current = queue.shift();
    for (const sem of plan) {
      for (const cid of sem) {
        if (dependents.has(cid)) continue;
        const course = getCourse(cid);
        if (course && course.prerequisites.includes(current)) {
          dependents.add(cid);
          queue.push(cid);
        }
      }
    }
  }

  return dependents;
}

/**
 * Performs a cascading update: when a course is removed from the plan,
 * all dependent courses are also removed. Returns the new plan.
 */
export function cascadeRemove(courseId, plan) {
  const dependents = getDependents(courseId, plan);
  dependents.add(courseId);

  return plan.map((sem) => sem.filter((id) => !dependents.has(id)));
}

/**
 * Computes degree progress: how many requirements are completed.
 */
export function computeProgress(plan, completedSemesters) {
  const completed = new Set();
  for (let i = 0; i < completedSemesters; i++) {
    if (plan[i]) plan[i].forEach((id) => completed.add(id));
  }

  let totalRequired = 0;
  let totalCompleted = 0;
  let totalCreditsCompleted = 0;
  let totalCreditsPlanned = 0;

  const groupProgress = degreeRequirements.map((req) => {
    const allRequired = [...req.requiredCourses];
    const total = allRequired.length + req.electivesNeeded;
    totalRequired += total;

    let done = 0;
    allRequired.forEach((id) => {
      if (completed.has(id)) done++;
    });

    // Count electives
    let electivesDone = 0;
    req.electivePool.forEach((id) => {
      if (completed.has(id)) electivesDone++;
    });
    done += Math.min(electivesDone, req.electivesNeeded);
    totalCompleted += done;

    return { name: req.name, total, completed: done, percent: total > 0 ? Math.round((done / total) * 100) : 100 };
  });

  // Credits
  for (const id of completed) {
    const c = getCourse(id);
    if (c) totalCreditsCompleted += c.credits;
  }
  const allPlanned = new Set(plan.flat());
  for (const id of allPlanned) {
    const c = getCourse(id);
    if (c) totalCreditsPlanned += c.credits;
  }

  return {
    groupProgress,
    totalRequired,
    totalCompleted,
    totalCreditsCompleted,
    totalCreditsPlanned,
    overallPercent: totalRequired > 0 ? Math.round((totalCompleted / totalRequired) * 100) : 0,
  };
}

/**
 * Auto-schedule: try to build an optimal plan that fits within maxSemesters.
 * Uses topological sort by prerequisites.
 */
export function autoSchedule(completedCourses = [], maxSemesters = 8) {
  // Gather all required courses
  const needed = new Set();
  degreeRequirements.forEach((req) => {
    req.requiredCourses.forEach((id) => needed.add(id));
    // Add first N from elective pool
    let added = 0;
    for (const id of req.electivePool) {
      if (added >= req.electivesNeeded) break;
      needed.add(id);
      added++;
    }
  });

  // Remove already completed
  completedCourses.forEach((id) => needed.delete(id));

  // Topological sort
  const sorted = [];
  const visited = new Set(completedCourses);
  const visiting = new Set();

  function visit(id) {
    if (visited.has(id)) return;
    if (visiting.has(id)) return; // cycle
    visiting.add(id);
    const course = getCourse(id);
    if (course) {
      course.prerequisites.forEach((pid) => {
        if (needed.has(pid)) visit(pid);
      });
    }
    visiting.delete(id);
    visited.add(id);
    sorted.push(id);
  }

  needed.forEach((id) => visit(id));

  // Place in semesters (max ~5 courses each, respecting semester availability)
  const plan = Array.from({ length: maxSemesters }, () => []);
  const placed = new Set(completedCourses);
  const MAX_PER_SEM = 5;

  for (const courseId of sorted) {
    for (let sem = 0; sem < maxSemesters; sem++) {
      if (plan[sem].length >= MAX_PER_SEM) continue;
      if (!isOfferedInSemester(courseId, sem)) continue;

      const course = getCourse(courseId);
      const prereqsMet = course.prerequisites.every((pid) => placed.has(pid));
      if (!prereqsMet) continue;

      plan[sem].push(courseId);
      placed.add(courseId);
      break;
    }
  }

  return plan;
}
