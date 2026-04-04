import React from "react";
import { getCourse, CATEGORY } from "../data/catalog";

function getIndicatorClass(category) {
  switch (category) {
    case CATEGORY.CS_CORE: return "indicator-core";
    case CATEGORY.MATH: return "indicator-math";
    case CATEGORY.SCIENCE: return "indicator-science";
    case CATEGORY.GEN_ED: return "indicator-gened";
    case CATEGORY.TECH_ELECTIVE: return "indicator-elective";
    case CATEGORY.FREE_ELECTIVE: return "indicator-free";
    default: return "indicator-core";
  }
}

export default function CourseCard({ courseId, semesterIndex, isCompleted, issues, onClick, onRemove }) {
  const course = getCourse(courseId);
  if (!course) return null;

  const hasIssue = issues && issues.length > 0;

  return (
    <div
      className={`course-card ${hasIssue ? "has-issue" : ""} ${isCompleted ? "is-completed" : ""}`}
      onClick={onClick}
      title={hasIssue ? issues.join("\n") : course.description}
    >
      <div className={`course-indicator ${getIndicatorClass(course.category)}`} />
      <div className="course-info">
        <div className="course-code">{course.code}</div>
        <div className="course-name">{course.name}</div>
      </div>
      <div className="course-credits">{course.credits}cr</div>
      <div className="course-actions">
        <button
          className="course-action-btn remove"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          title="Remove (cascading)"
        >
          &#x2715;
        </button>
      </div>
    </div>
  );
}
