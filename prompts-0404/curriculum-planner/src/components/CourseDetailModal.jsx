import React from "react";
import { getCourse, CATEGORY, semesterLabel } from "../data/catalog";
import { getSubstitutions, arePrerequisitesMet, areCorequisitesMet, getDependents } from "../utils/planEngine";

function getBadgeClass(category) {
  switch (category) {
    case CATEGORY.CS_CORE: return "badge-core";
    case CATEGORY.MATH: return "badge-math";
    case CATEGORY.SCIENCE: return "badge-science";
    case CATEGORY.GEN_ED: return "badge-gened";
    case CATEGORY.TECH_ELECTIVE: return "badge-elective";
    default: return "badge-core";
  }
}

export default function CourseDetailModal({ courseId, semesterIndex, plan, onClose, onSubstitute }) {
  const course = getCourse(courseId);
  if (!course) return null;

  const prereqCheck = arePrerequisitesMet(courseId, semesterIndex, plan);
  const coreqCheck = areCorequisitesMet(courseId, semesterIndex, plan);
  const dependents = getDependents(courseId, plan);
  const substitutions = getSubstitutions(courseId);

  // Find which semester this course is actually in
  let actualSemIdx = semesterIndex;
  for (let i = 0; i < plan.length; i++) {
    if (plan[i].includes(courseId)) {
      actualSemIdx = i;
      break;
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal fade-in" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>
          &#x2715;
        </button>

        <div className="course-detail-header">
          <span className={`course-detail-badge ${getBadgeClass(course.category)}`}>{course.category}</span>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{course.credits} credits</span>
        </div>

        <h2>{course.code} — {course.name}</h2>
        <p className="modal-subtitle">{course.description}</p>

        <div className="detail-row">
          <span className="detail-label">Semester Offered</span>
          <span className="detail-value">{course.semesters.join(", ")}</span>
        </div>

        <div className="detail-row">
          <span className="detail-label">Placed In</span>
          <span className="detail-value">{semesterLabel(actualSemIdx)}</span>
        </div>

        <div className="detail-row">
          <span className="detail-label">Prerequisites</span>
          <span className="detail-value">
            {course.prerequisites.length === 0
              ? "None"
              : course.prerequisites.map((id) => getCourse(id)?.code || id).join(", ")}
          </span>
        </div>

        {course.corequisites && course.corequisites.length > 0 && (
          <div className="detail-row">
            <span className="detail-label">Co-requisites</span>
            <span className="detail-value">
              {course.corequisites.map((id) => getCourse(id)?.code || id).join(", ")}
            </span>
          </div>
        )}

        <div className="detail-row">
          <span className="detail-label">Prerequisites Met?</span>
          <span className="detail-value" style={{ color: prereqCheck.met ? "var(--accent-green)" : "var(--accent-red)" }}>
            {prereqCheck.met ? "Yes" : `No — missing: ${prereqCheck.missing.map((id) => getCourse(id)?.code || id).join(", ")}`}
          </span>
        </div>

        {dependents.size > 0 && (
          <div className="detail-row">
            <span className="detail-label">Dependents</span>
            <span className="detail-value" style={{ fontSize: 12 }}>
              {[...dependents].map((id) => getCourse(id)?.code || id).join(", ")}
            </span>
          </div>
        )}

        {substitutions.length > 0 && (
          <div className="substitution-list">
            <h4>Available Substitutions</h4>
            <p style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 10 }}>
              Selecting a substitution will replace this course and update the plan accordingly.
            </p>
            {substitutions.map((sub) => (
              <div key={sub.id} className="substitution-option">
                <div>
                  <div style={{ fontWeight: 600, fontSize: 13 }}>{sub.code}</div>
                  <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>{sub.name}</div>
                </div>
                <button className="sub-btn" onClick={() => onSubstitute(courseId, sub.id, actualSemIdx)}>
                  Swap
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
