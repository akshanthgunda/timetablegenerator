import streamlit as st
import pandas as pd
import random
from collections import defaultdict
import numpy as np

st.set_page_config(layout="wide")

def main():
    st.title("🧠 Smart School Timetable Generator")
    
    # --------- Input Tables ---------
    st.header("1. Teacher Information")
    teacher_df = st.data_editor(
        pd.DataFrame({
            "Teacher": ["Ms. Smith", "Mr. Patel"],
            "Subjects": ["English", "Math, Science"],
            "Classes": ["6A, 6B", "6A, 6C, 7A"],
            "Available Days": ["Mon, Tue, Wed", "Mon, Thu, Fri"],
            "Available Periods": ["1,2,3", "2,3,4,5"]
        }),
        num_rows="dynamic",
        key="teacher_editor"
    )

    st.header("2. Subject–Class Frequency Table")
    subject_df = st.data_editor(
        pd.DataFrame({
            "Class": ["6A,6B,6C,7A,7B"],
            "Subject": ["English"],
            "Periods_Per_Week": [6]
        }),
        num_rows="dynamic",
        key="subject_editor"
    )

    st.header("3. Options")
    periods_per_day = st.number_input("Periods per Day", 1, 10, 6)
    working_days = st.multiselect("Working Days", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], default=["Mon", "Tue", "Wed", "Thu", "Fri"])
    
    spacing_options = st.expander("Spacing and Efficiency Options", expanded=True)
    with spacing_options:
        min_days_between_same_subject = st.slider(
            "Minimum days between same subject (when possible)", 
            0, 3, 1,
            help="Try to maintain at least this many days between consecutive classes of the same subject"
        )
        encourage_blocks = st.checkbox(
            "Encourage block periods for subjects with multiple periods per week", 
            value=False,
            help="When enabled, the system will try to schedule consecutive periods for subjects with higher frequency"
        )
        max_consecutive_periods = st.slider(
            "Maximum consecutive periods for same subject", 
            1, 3, 2,
            help="Maximum number of consecutive periods allowed for the same subject"
        )
        balance_daily_load = st.checkbox(
            "Balance teacher daily workload", 
            value=True,
            help="Try to distribute teacher's classes evenly across their available days"
        )

    if st.button("Generate Timetable"):
        # --------- Step 1: Expand subject table ---------
        expanded_subjects = []
        for _, row in subject_df.iterrows():
            subject = row["Subject"].strip()
            try:
                periods = int(row["Periods_Per_Week"])
            except:
                st.error(f"Invalid number of periods: {row['Periods_Per_Week']}")
                return
            class_list = [c.strip() for c in str(row["Class"]).split(",") if c.strip()]
            for cls in class_list:
                expanded_subjects.append({
                    "Class": cls,
                    "Subject": subject,
                    "Periods_Per_Week": periods
                })
        subject_df_expanded = pd.DataFrame(expanded_subjects)

        # --------- Step 2: Prepare timetable structure ---------
        classes = subject_df_expanded["Class"].unique().tolist()
        timetable = {
            cls: pd.DataFrame('', index=working_days, columns=[f"P{p+1}" for p in range(int(periods_per_day))])
            for cls in classes
        }
        
        # Track when subjects were last scheduled
        last_scheduled = {
            cls: defaultdict(lambda: -999) for cls in classes
        }
        
        # Track teacher daily assignments for load balancing
        teacher_daily_load = defaultdict(lambda: defaultdict(int))

        # --------- Step 3: Process teacher assignments ---------
        teacher_assignments = []
        for _, row in teacher_df.iterrows():
            subjects = [s.strip() for s in str(row["Subjects"]).split(",")]
            classes_ = [c.strip() for c in str(row["Classes"]).split(",")]
            days = [d.strip() for d in str(row["Available Days"]).split(",")]
            periods = [int(p.strip()) for p in str(row["Available Periods"]).split(",") if p.strip().isdigit()]
            for subj in subjects:
                for cls in classes_:
                    teacher_assignments.append({
                        "teacher": row["Teacher"],
                        "subject": subj,
                        "class": cls,
                        "available_days": days,
                        "available_periods": periods
                    })

        def is_available(teacher, day, period):
            for t in teacher_assignments:
                if t["teacher"] == teacher:
                    return day in t["available_days"] and period in t["available_periods"]
            return False

        def already_assigned(timetable, day, period, teacher):
            for df in timetable.values():
                if df.at[day, f"P{period}"] and teacher in df.at[day, f"P{period}"]:
                    return True
            return False
        
        def get_day_index(day):
            return working_days.index(day)

        # --------- Step 4: Sort subjects by priority ---------
        # Give priority to subjects with more periods needed
        subjects_to_schedule = []
        for _, row in subject_df_expanded.iterrows():
            subjects_to_schedule.append({
                "class": row["Class"],
                "subject": row["Subject"],
                "periods_needed": int(row["Periods_Per_Week"])
            })
        
        # Sort by periods needed (descending) to schedule highest-demand subjects first
        subjects_to_schedule.sort(key=lambda x: x["periods_needed"], reverse=True)

        # --------- Step 5: Assign periods with improved spacing ---------
        for subject_info in subjects_to_schedule:
            cls = subject_info["class"]
            subject = subject_info["subject"]
            periods_needed = subject_info["periods_needed"]
            
            # Get eligible teachers for this subject and class
            eligible_teachers = [t for t in teacher_assignments 
                                if t["subject"] == subject and t["class"] == cls]
            
            if not eligible_teachers:
                st.warning(f"⚠️ No eligible teachers found for {subject} in {cls}.")
                continue
                
            df = timetable[cls]
            
            # Distribution strategy
            # For optimal spacing, we want to spread across specific days
            ideal_gap = max(1, len(working_days) // periods_needed)
            min_gap = min(min_days_between_same_subject, ideal_gap)
            
            # Track assigned periods
            assigned_count = 0
            
            # Step 1: Create a scoring function for day selection
            def score_day_for_subject(day, subject, cls):
                day_idx = get_day_index(day)
                score = 0
                
                # Prefer days that maintain minimum spacing from last scheduled day
                last_day_idx = get_day_index(last_scheduled[cls][subject]) if last_scheduled[cls][subject] in working_days else -999
                days_since_last = day_idx - last_day_idx if last_day_idx != -999 else 999
                
                if days_since_last < min_gap:
                    score -= 100  # Strong penalty for violating min spacing
                else:
                    score += days_since_last  # Prefer maximum spacing
                
                # Consider teacher load balancing
                if balance_daily_load:
                    for teacher in [t["teacher"] for t in eligible_teachers]:
                        score -= teacher_daily_load[teacher][day] * 10  # Penalize days where teacher already has many classes
                
                # Check availability of periods on this day
                available_periods = sum(1 for p in range(1, periods_per_day + 1) 
                                      if df.at[day, f"P{p}"] == '' and 
                                      any(is_available(t["teacher"], day, p) for t in eligible_teachers))
                
                score += available_periods * 5  # Prefer days with more available periods
                
                # Randomize slightly to avoid identical patterns
                score += random.random() * 2
                
                return score
            
            # Iteration strategy for efficient assignment
            attempt_count = 0
            max_attempts = periods_needed * 5  # Limit attempts to avoid infinite loops
            
            while assigned_count < periods_needed and attempt_count < max_attempts:
                attempt_count += 1
                
                # Score all days and choose the best
                day_scores = [(day, score_day_for_subject(day, subject, cls)) for day in working_days]
                day_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by score descending
                
                for day, _ in day_scores:
                    # Determine how many periods to assign on this day
                    max_periods_today = 1
                    if encourage_blocks and periods_needed - assigned_count >= 2:
                        max_periods_today = min(max_consecutive_periods, periods_needed - assigned_count)
                    
                    # Score periods for this day
                    period_scores = []
                    for period in range(1, periods_per_day + 1):
                        col = f"P{period}"
                        if df.at[day, col] == '':
                            # Check for teachers available during this period
                            available_teachers = [
                                t for t in eligible_teachers 
                                if is_available(t["teacher"], day, period) and 
                                not already_assigned(timetable, day, period, t["teacher"])
                            ]
                            
                            if available_teachers:
                                # Score this period
                                score = 0
                                
                                # Prefer periods that make blocks if enabled
                                if encourage_blocks:
                                    # Check if previous period has same subject
                                    if period > 1 and f"{subject}\n" in df.at[day, f"P{period-1}"]:
                                        score += 50
                                    
                                    # Check if next period is available
                                    if (period < periods_per_day and 
                                        df.at[day, f"P{period+1}"] == '' and
                                        any(is_available(t["teacher"], day, period+1) and 
                                            not already_assigned(timetable, day, period+1, t["teacher"]) 
                                            for t in eligible_teachers)):
                                        score += 30
                                        
                                # Add some randomness for variety
                                score += random.random() * 5
                                
                                period_scores.append((period, score, available_teachers))
                    
                    # Skip day if no periods available
                    if not period_scores:
                        continue
                    
                    # Sort periods by score
                    period_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Assign the best periods up to max_periods_today
                    assigned_today = 0
                    for period, _, teachers in period_scores:
                        if assigned_today >= max_periods_today or assigned_count >= periods_needed:
                            break
                            
                        col = f"P{period}"
                        
                        # Choose teacher with least load for this day
                        teachers.sort(key=lambda t: teacher_daily_load[t["teacher"]][day])
                        chosen_teacher = teachers[0]["teacher"]
                        
                        # Make the assignment
                        df.at[day, col] = f"{subject}\n({chosen_teacher})"
                        assigned_count += 1
                        assigned_today += 1
                        
                        # Update tracking variables
                        last_scheduled[cls][subject] = day
                        teacher_daily_load[chosen_teacher][day] += 1
                    
                    if assigned_today > 0:
                        break  # Move to next day selection if we assigned periods
            
            if assigned_count < periods_needed:
                st.warning(f"⚠️ Only {assigned_count}/{periods_needed} periods were scheduled for {subject} in {cls} after optimization.")

        # --------- Step 6: Analyze and score the timetable ---------
        total_score = 0
        spacing_violations = 0
        teacher_imbalance = 0
        unscheduled_periods = 0
        
        # Calculate total assigned periods
        total_assigned = sum(sum(1 for val in df.values.flatten() if val != '') for df in timetable.values())
        
        # Calculate total required periods
        total_required = sum(row["Periods_Per_Week"] for _, row in subject_df_expanded.iterrows())
        
        # Calculate spacing violations
        for cls in classes:
            subject_days = defaultdict(list)
            df = timetable[cls]
            
            for day in working_days:
                day_idx = working_days.index(day)
                for period in range(1, periods_per_day + 1):
                    cell = df.at[day, f"P{period}"]
                    if cell:
                        subject = cell.split("\n")[0]
                        subject_days[subject].append(day_idx)
            
            # Check spacing between days of the same subject
            for subject, days in subject_days.items():
                days.sort()
                for i in range(1, len(days)):
                    if days[i] - days[i-1] < min_days_between_same_subject:
                        spacing_violations += 1
        
        # Calculate teacher workload imbalance
        for teacher, days in teacher_daily_load.items():
            if days:  # Skip empty records
                loads = list(days.values())
                if loads:
                    teacher_imbalance += np.std(loads)  # Standard deviation of teacher's daily loads
        
        # Calculate unscheduled periods
        unscheduled_periods = total_required - total_assigned
        
        # Calculate final score (higher is better)
        efficiency_score = 100
        if total_required > 0:
            efficiency_score -= (unscheduled_periods / total_required) * 50
        efficiency_score -= spacing_violations * 2
        efficiency_score -= teacher_imbalance * 5
        
        st.success(f"✅ Timetable Generation Complete. Efficiency Score: {efficiency_score:.1f}/100")
        
        # Show efficiency metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Scheduled Periods", f"{total_assigned}/{total_required}", 
                           f"{(total_assigned/total_required*100):.1f}%" if total_required > 0 else "N/A")
        metrics_col2.metric("Spacing Violations", str(spacing_violations))
        metrics_col3.metric("Teacher Load Balance", f"{teacher_imbalance:.2f} σ")

        # --------- Display Results ---------
        st.subheader("📊 Timetable Results")
        
        for cls, df in timetable.items():
            with st.expander(f"📘 Timetable for Class {cls}", expanded=True):
                # Highlight cells for better visualization
                def color_cells(val):
                    if not val:
                        return 'background-color: #f0f2f6'
                    
                    subject = val.split('\n')[0]
                    # Create a deterministic color based on subject name
                    color_hash = hash(subject) % 100000
                    hue = color_hash % 360
                    return f'background-color: hsla({hue}, 70%, 85%, 0.8); color: black'
                
                styled_df = df.style.applymap(color_cells)
                st.dataframe(styled_df, use_container_width=True)
        
        # --------- Teacher Schedules ---------
        st.subheader("👨‍🏫 Teacher Schedules")
        for teacher in {t["teacher"] for t in teacher_assignments}:
            with st.expander(f"Schedule for {teacher}"):
                # Create a new DataFrame for this teacher
                teacher_schedule = pd.DataFrame('', index=working_days, 
                                            columns=[f"P{p+1}" for p in range(int(periods_per_day))])
                
                # Fill in the teacher's assignments
                for cls, df in timetable.items():
                    for day in working_days:
                        for period in range(1, periods_per_day + 1):
                            cell = df.at[day, f"P{period}"]
                            if cell and teacher in cell:
                                subject = cell.split('\n')[0]
                                teacher_schedule.at[day, f"P{period}"] = f"{subject}\nClass {cls}"
                
                # Display with styling
                def highlight_teacher_cells(val):
                    if not val:
                        return 'background-color: #f0f2f6'
                    return 'background-color: #c6e5f9; color: black'
                
                st.dataframe(teacher_schedule.style.applymap(highlight_teacher_cells), use_container_width=True)
                
                # Show daily load
                days_worked = sum(1 for day in working_days if any(teacher_schedule.loc[day]))
                total_periods = sum(sum(1 for cell in teacher_schedule.loc[day] if cell) for day in working_days)
                st.text(f"Total workload: {total_periods} periods across {days_worked} days")
                
                if days_worked > 0:
                    avg_load = total_periods / days_worked
                    st.text(f"Average daily load: {avg_load:.2f} periods per day")

if __name__ == "__main__":
    main()
