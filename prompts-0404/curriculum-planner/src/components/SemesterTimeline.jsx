import React from "react";
import CourseCard from "./CourseCard";
import { semesterLabel, getCourse } from "../data/catalog";

export default function SemesterTimeline({
  plan,
  issues,
  completedSemesters,
  onCourseClick,
  onRemoveCourse,
  onAddCourse,
  onToggleComplete,
}) {
  return (
    <div className="timeline">
      {plan.map((semCourses, semIdx) => {
        const isCompleted = semIdx < completedSemesters;
        const isInProgress = semIdx === completedSemesters;
        const totalCredits = semCourses.reduce((sum, id) => {
          const c = getCourse(id);
          return sum + (c ? c.credits : 0);
        }, 0);

        const statusClass = isCompleted ? "completed" : isInProgress ? "in-progress" : "";

        return (
          <div className="semester-card fade-in" key={semIdx} style={{ animationDelay: `${semIdx * 50}ms` }}>
            <div className={`semester-header ${statusClass}`}>
              <div>
                <div className="semester-name">{semesterLabel(semIdx)}</div>
                <div className="semester-credits">{totalCredits} credits</div>
              </div>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span
                  className={`semester-status ${
                    isCompleted ? "status-completed" : isInProgress ? "status-in-progress" : "status-planned"
                  }`}
                >
                  {isCompleted ? "Done" : isInProgress ? "Current" : "Planned"}
                </span>
                <button
                  className="btn btn-secondary"
                  style={{ fontSize: 11, padding: "2px 8px" }}
                  onClick={() => onToggleComplete(semIdx)}
                  title={isCompleted ? "Mark as not completed" : "Mark as completed"}
                >
                  {isCompleted ? "Undo" : "Complete"}
                </button>
              </div>
            </div>
            <div className="semester-courses">
              {semCourses.map((courseId) => (
                <CourseCard
                  key={courseId}
                  courseId={courseId}
                  semesterIndex={semIdx}
                  isCompleted={isCompleted}
                  issues={issues[semIdx]?.[courseId]}
                  onClick={() => onCourseClick(courseId, semIdx)}
                  onRemove={() => onRemoveCourse(semIdx, courseId)}
                />
              ))}
              <button className="add-course-btn" onClick={() => onAddCourse(semIdx)}>
                + Add Course
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
