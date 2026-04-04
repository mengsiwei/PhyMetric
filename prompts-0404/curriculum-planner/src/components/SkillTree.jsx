import React, { useMemo } from "react";
import { courses, getCourse, CATEGORY } from "../data/catalog";

/**
 * Skill Tree: Visualizes prerequisite chains as a dependency graph.
 * Courses are laid out in "levels" based on their prerequisite depth.
 */

function getPrereqDepth(courseId, memo = {}) {
  if (memo[courseId] !== undefined) return memo[courseId];
  const course = getCourse(courseId);
  if (!course || course.prerequisites.length === 0) {
    memo[courseId] = 0;
    return 0;
  }
  const maxParent = Math.max(...course.prerequisites.map((pid) => getPrereqDepth(pid, memo)));
  memo[courseId] = maxParent + 1;
  return memo[courseId];
}

export default function SkillTree({ plan, completedSemesters, onCourseClick }) {
  const completedSet = useMemo(() => {
    const s = new Set();
    for (let i = 0; i < completedSemesters; i++) {
      plan[i]?.forEach((id) => s.add(id));
    }
    return s;
  }, [plan, completedSemesters]);

  const inProgressSet = useMemo(() => {
    return new Set(plan[completedSemesters] || []);
  }, [plan, completedSemesters]);

  const plannedSet = useMemo(() => new Set(plan.flat()), [plan]);

  // Build levels
  const levels = useMemo(() => {
    const memo = {};
    const mainCourses = courses.filter(
      (c) =>
        !c.hidden &&
        (c.category === CATEGORY.CS_CORE ||
          c.category === CATEGORY.MATH ||
          c.category === CATEGORY.TECH_ELECTIVE)
    );

    mainCourses.forEach((c) => getPrereqDepth(c.id, memo));

    const maxLevel = Math.max(...Object.values(memo), 0);
    const result = [];
    for (let l = 0; l <= maxLevel; l++) {
      result.push(mainCourses.filter((c) => memo[c.id] === l));
    }
    return result;
  }, []);

  function getNodeClass(id) {
    if (completedSet.has(id)) return "completed";
    if (inProgressSet.has(id)) return "in-progress";
    if (!plannedSet.has(id)) return "locked";
    return "";
  }

  return (
    <div className="skill-tree">
      {levels.map((levelCourses, levelIdx) => (
        <div key={levelIdx}>
          <div
            style={{
              fontSize: 11,
              fontWeight: 700,
              color: "var(--text-muted)",
              textTransform: "uppercase",
              marginBottom: 8,
              letterSpacing: "0.05em",
            }}
          >
            Level {levelIdx}
            {levelIdx === 0 && " — Entry"}
            {levelIdx === levels.length - 1 && " — Capstone"}
          </div>
          <div className="skill-tree-row">
            {levelCourses.map((course, i) => (
              <React.Fragment key={course.id}>
                <div
                  className={`skill-node ${getNodeClass(course.id)}`}
                  onClick={() => onCourseClick(course.id)}
                >
                  {completedSet.has(course.id) && <div className="skill-node-check">&#10003;</div>}
                  <div className="skill-node-code">{course.code}</div>
                  <div className="skill-node-name">{course.name}</div>
                </div>
                {i < levelCourses.length - 1 && (
                  <div
                    className={`skill-connector ${
                      completedSet.has(levelCourses[i].id) && completedSet.has(levelCourses[i + 1]?.id)
                        ? "active"
                        : ""
                    }`}
                  />
                )}
              </React.Fragment>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
