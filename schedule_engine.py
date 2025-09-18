from typing import Dict, List, Tuple, Set, Optional
def print_validation_report(validation_report: Dict):
    """
    Print a comprehensive validation report with accuracy metrics.
    
    Args:
        validation_report: Validation results dictionary
    """
    print("\n" + "="*70)
    print("SCHEDULE VALIDATION REPORT")
    print("="*70)
    
    # Overall Results
    accuracy = validation_report.get("overall_accuracy", 0)
    passed = validation_report.get("passed_constraints", 0)
    total = validation_report.get("total_constraints", 0)
    
    print(f"\n📊 OVERALL ACCURACY: {accuracy:.1f}% ({passed}/{total} constraints passed)")
    
    if accuracy >= 95:
        print("🟢 EXCELLENT - Schedule meets all critical requirements")
    elif accuracy >= 80:
        print("🟡 GOOD - Minor issues that may need attention")
    else:
        print("🔴 POOR - Significant problems found")
    
    # Constraint Details
    print(f"\n📋 CONSTRAINT VALIDATION:")
    print("-" * 50)
    
    for constraint_name, details in validation_report.get("constraint_details", {}).items():
        status = "✅ PASS" if details["passed"] else ("❌ FAIL" if details["critical"] else "⚠️  WARN")
        print(f"{status} {constraint_name}")
        if details["details"]:
            print(f"     {details['details']}")
    
    # Performance Metrics
    metrics = validation_report.get("performance_metrics", {})
    if metrics:
        print(f"\n📈 PERFORMANCE METRICS:")
        print("-" * 50)
        print(f"Assignment Rate:     {metrics.get('assignment_rate', 0):.1f}% ({metrics.get('total_trips_assigned', 0)}/{metrics.get('total_trips_expected', 0)} trips)")
        print(f"Load Balance Score:  {metrics.get('load_balance_score', 0):.1f}%")
        print(f"Train Utilization:   {metrics.get('utilization_rate', 0):.1f}% ({metrics.get('trains_utilized', 0)}/{len(validation_report.get('schedule', {}))} trains)")
        print(f"Trips per Train:     {metrics.get('min_trips_per_train', 0)} - {metrics.get('max_trips_per_train', 0)} (avg: {metrics.get('avg_trips_per_train', 0):.1f})")
        print(f"Load Std Deviation:  {metrics.get('std_dev_trips', 0):.2f}")
    
    # Issues Summary
    failed = validation_report.get("failed_constraints", [])
    warnings = validation_report.get("warnings", [])
    
    if failed:
        print(f"\n🚨 CRITICAL ISSUES ({len(failed)}):")
        print("-" * 50)
        for issue in failed:
            print(f"   • {issue}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        print("-" * 50)
        for warning in warnings:
            print(f"   • {warning}")
    
    if not failed and not warnings:
        print(f"\n🎉 NO ISSUES FOUND - Schedule is optimal!")
    
    print("="*70)

def calculate_schedule_quality_score(validation_report: Dict) -> Dict[str, float]:
    """
    Calculate comprehensive quality metrics for the schedule.
    
    Args:
        validation_report: Validation results
        
    Returns:
        Dictionary of quality scores
    """
    metrics = validation_report.get("performance_metrics", {})
    
    # Constraint Compliance Score (0-100)
    constraint_score = validation_report.get("overall_accuracy", 0)
    
    # Load Balance Score (0-100) - higher is better (more balanced)
    load_balance_score = metrics.get("load_balance_score", 0)
    
    # Utilization Score (0-100) - percentage of trains used
    utilization_score = metrics.get("utilization_rate", 0)
    
    # Assignment Completeness Score (0-100) - percentage of trips assigned
    assignment_score = metrics.get("assignment_rate", 0)
    
    # Efficiency Score - based on standard deviation (lower std dev = higher score)
    std_dev = metrics.get("std_dev_trips", float('inf'))
    max_possible_std = metrics.get("max_trips_per_train", 1)
    efficiency_score = max(0, 100 - (std_dev / max_possible_std * 100)) if max_possible_std > 0 else 0
    
    # Overall Quality Score (weighted average)
    weights = {
        "constraint_compliance": 0.4,    # Most important - must meet constraints
        "load_balance": 0.25,           # Even distribution important
        "utilization": 0.15,            # Use available resources
        "assignment_completeness": 0.15, # Complete all assignments
        "efficiency": 0.05              # Minimize variance
    }
    
    overall_score = (
        constraint_score * weights["constraint_compliance"] +
        load_balance_score * weights["load_balance"] + 
        utilization_score * weights["utilization"] +
        assignment_score * weights["assignment_completeness"] +
        efficiency_score * weights["efficiency"]
    )
    
    return {
        "overall_quality": overall_score,
        "constraint_compliance": constraint_score,
        "load_balance": load_balance_score,
        "utilization": utilization_score,
        "assignment_completeness": assignment_score,
        "efficiency": efficiency_score,
        "grade": get_quality_grade(overall_score)
    }

def get_quality_grade(score: float) -> str:
    """Convert quality score to letter grade."""
    if score >= 90: return "A+"
    elif score >= 85: return "A"
    elif score >= 80: return "A-"
    elif score >= 75: return "B+"
    elif score >= 70: return "B"
    elif score >= 65: return "B-"
    elif score >= 60: return "C+"
    elif score >= 55: return "C"
    elif score >= 50: return "C-"
    else: return "F"#!/usr/bin/env python3
"""
Train Trip Scheduler using Google OR-Tools CP-SAT Solver

This program schedules train trips with constraints for timing, continuity,
maintenance requirements, and load balancing.

Requirements:
- pip install ortools

Author: AI Assistant
Date: September 2025
"""

import json
import time
from ortools.sat.python import cp_model
from collections import defaultdict, Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
NUM_TRAINS = 25
MAINTENANCE_TRAINS = {"Train_1", "Train_5", "Train_10", "Train_20"}
HUB_STOPS = {"ALVA", "MUTT", "CGPP", "PARV", "VYTA", "TPHT"}
STATION_ORDER = [
    "ALVA","PNCU","CPPY","ATTK","MUTT","KLMT","CCUV","PDPM","EDAP","CGPP",
    "PARV","JLSD","KALR","TNHL","MGRD","MACE","ERSH","KVTR","EMKM","VYTA",
    "THYK","PETT","VAKK","SNJN","TPHT"
]

def time_to_seconds(time_str: str) -> int:
    """
    Convert time string (HH:MM:SS) to seconds since midnight.
    
    Args:
        time_str: Time in format "HH:MM:SS"
    
    Returns:
        int: Seconds since midnight
    """
    try:
        hours, minutes, seconds = map(int, time_str.split(':'))
        return hours * 3600 + minutes * 60 + seconds
    except (ValueError, AttributeError):
        logger.error(f"Invalid time format: {time_str}")
        raise ValueError(f"Invalid time format: {time_str}")

def seconds_to_time(seconds: int) -> str:
    """
    Convert seconds since midnight to time string (HH:MM:SS).
    
    Args:
        seconds: Seconds since midnight
    
    Returns:
        str: Time in format "HH:MM:SS"
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def load_trips(filename: str) -> List[Dict]:
    """
    Load trips from JSON file.
    
    Args:
        filename: Path to JSON file containing trips
    
    Returns:
        List of trip dictionaries
    """
    try:
        with open(filename, 'r') as f:
            trips = json.load(f)
        logger.info(f"Loaded {len(trips)} trips from {filename}")
        return trips
    except FileNotFoundError:
        logger.error(f"File {filename} not found")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filename}: {e}")
        raise

def calculate_trip_duration(trip: Dict) -> Tuple[int, int]:
    """
    Calculate start and end times for a trip in seconds.
    
    Args:
        trip: Trip dictionary with stops
    
    Returns:
        Tuple of (start_time_seconds, end_time_seconds)
    """
    if not trip.get('stops'):
        raise ValueError(f"Trip {trip.get('trip_id', 'unknown')} has no stops")
    
    first_stop = trip['stops'][0]
    last_stop = trip['stops'][-1]
    
    start_time = time_to_seconds(first_stop['departure_time'])
    end_time = time_to_seconds(last_stop['arrival_time'])
    
    return start_time, end_time

def get_trip_endpoints(trip: Dict) -> Tuple[str, str]:
    """
    Get the start and end stop IDs for a trip.
    
    Args:
        trip: Trip dictionary with stops
    
    Returns:
        Tuple of (start_stop_id, end_stop_id)
    """
    if not trip.get('stops'):
        raise ValueError(f"Trip {trip.get('trip_id', 'unknown')} has no stops")
    
    start_stop = trip['stops'][0]['stop_id']
    end_stop = trip['stops'][-1]['stop_id']
    
    return start_stop, end_stop

def calculate_travel_time(start_stop: str, end_stop: str, station_order: List[str]) -> int:
    """
    Calculate minimum travel time between two stops on a linear route.
    ALVA and TPHT are endpoints - trains cannot teleport between them.
    
    Args:
        start_stop: Starting stop ID
        end_stop: Ending stop ID
        station_order: Ordered list of stations (ALVA to TPHT)
    
    Returns:
        int: Travel time in seconds, or -1 if impossible connection
    """
    try:
        start_idx = station_order.index(start_stop)
        end_idx = station_order.index(end_stop)
        
        # On a linear route, travel time is always based on distance
        distance = abs(end_idx - start_idx)
        return distance * 120  # 2 minutes per station
    except ValueError:
        # If stops not in order, assume reasonable travel time
        logger.warning(f"Stops {start_stop} or {end_stop} not found in station order")
        return 600  # 10 minutes default

def trips_can_be_consecutive(trip1: Dict, trip2: Dict, station_order: List[str]) -> bool:
    """
    Check if trip2 can follow trip1 on the same train.
    Considers linear route constraints - trains cannot teleport between endpoints.
    
    Args:
        trip1: First trip
        trip2: Second trip
        station_order: Ordered list of stations (ALVA to TPHT)
    
    Returns:
        bool: True if trips can be consecutive
    """
    _, trip1_end_stop = get_trip_endpoints(trip1)
    trip2_start_stop, _ = get_trip_endpoints(trip2)
    
    _, trip1_end_time = calculate_trip_duration(trip1)
    trip2_start_time, _ = calculate_trip_duration(trip2)
    
    # Check if there's enough time between trips
    travel_time = calculate_travel_time(trip1_end_stop, trip2_start_stop, station_order)
    buffer_time = 300  # 5-minute buffer
    
    # Check time constraint
    time_feasible = trip2_start_time >= trip1_end_time + travel_time + buffer_time
    
    # Additional check for linear route: if trip1 ends at one endpoint 
    # and trip2 starts at the other endpoint, it's not feasible
    # (train would need to travel the entire length of the line)
    endpoints = {station_order[0], station_order[-1]}  # ALVA and TPHT
    
    if trip1_end_stop in endpoints and trip2_start_stop in endpoints:
        if trip1_end_stop != trip2_start_stop:
            # Train is at one endpoint and needs to start at the other endpoint
            # This requires traveling the full length of the line
            full_line_travel_time = len(station_order) * 120  # Full line traversal
            time_feasible = trip2_start_time >= trip1_end_time + full_line_travel_time + buffer_time
    
    return time_feasible

class TrainScheduler:
    """
    Train scheduling solver using OR-Tools CP-SAT.
    """
    
    def __init__(self, trips: List[Dict], num_trains: int = NUM_TRAINS,
                 maintenance_trains: Set[str] = MAINTENANCE_TRAINS,
                 hub_stops: Set[str] = HUB_STOPS,
                 station_order: List[str] = STATION_ORDER):
        """
        Initialize the scheduler.
        
        Args:
            trips: List of trip dictionaries
            num_trains: Number of available trains
            maintenance_trains: Set of train IDs requiring maintenance
            hub_stops: Set of stop IDs where maintenance can occur
            station_order: Ordered list of stations for distance calculation
        """
        self.trips = trips
        self.num_trains = num_trains
        self.maintenance_trains = maintenance_trains
        self.hub_stops = hub_stops
        self.station_order = station_order
        
        # Generate train names
        self.train_names = [f"Train_{i+1}" for i in range(num_trains)]
        
        # Pre-calculate trip compatibility
        self._calculate_trip_compatibility()
        
        logger.info(f"Initialized scheduler with {len(trips)} trips and {num_trains} trains")
        logger.info(f"Maintenance trains: {maintenance_trains}")
        logger.info(f"Hub stops: {hub_stops}")
    
    def _calculate_trip_compatibility(self):
        """Pre-calculate which trips can follow each other."""
        self.compatible_trips = {}
        
        for i, trip1 in enumerate(self.trips):
            self.compatible_trips[i] = []
            for j, trip2 in enumerate(self.trips):
                if i != j and trips_can_be_consecutive(trip1, trip2, self.station_order):
                    self.compatible_trips[i].append(j)
        
        logger.info("Calculated trip compatibility matrix")
    
    def solve(self, time_limit_seconds: int = 120) -> Optional[Dict[str, List[str]]]:
        """
        Solve the train scheduling problem.
        
        Args:
            time_limit_seconds: Maximum solving time
        
        Returns:
            Dictionary mapping train names to lists of trip IDs, or None if no solution
        """
        model = cp_model.CpModel()
        
        # Decision variables: assignment[trip_idx][train_idx] = 1 if trip assigned to train
        assignment = {}
        for i in range(len(self.trips)):
            assignment[i] = {}
            for j in range(self.num_trains):
                assignment[i][j] = model.NewBoolVar(f'assign_trip_{i}_to_train_{j}')
        
        # Constraint 1: Each trip assigned to exactly one train
        for i in range(len(self.trips)):
            model.AddExactlyOne([assignment[i][j] for j in range(self.num_trains)])
        
        # Constraint 2 & 3: No overlapping trips and continuity on linear route
        for train_idx in range(self.num_trains):
            # For each pair of trips, check if they can be on the same train
            for i in range(len(self.trips)):
                for j in range(len(self.trips)):
                    if i != j:
                        start_time_i, end_time_i = calculate_trip_duration(self.trips[i])
                        start_time_j, end_time_j = calculate_trip_duration(self.trips[j])
                        
                        # Check for time overlap - overlapping trips cannot be on same train
                        times_overlap = not (end_time_i <= start_time_j or end_time_j <= start_time_i)
                        
                        if times_overlap:
                            model.AddBoolOr([
                                assignment[i][train_idx].Not(),
                                assignment[j][train_idx].Not()
                            ])
                        
                        # Check linear route continuity constraint
                        # If trip i ends before trip j starts, check if j can follow i
                        elif end_time_i < start_time_j:
                            if not trips_can_be_consecutive(self.trips[i], self.trips[j], self.station_order):
                                # Trip j cannot follow trip i due to linear route constraints
                                # So they cannot both be assigned to this train with i before j
                                model.AddBoolOr([
                                    assignment[i][train_idx].Not(),
                                    assignment[j][train_idx].Not()
                                ])
                        
                        # Similarly, if trip j ends before trip i starts
                        elif end_time_j < start_time_i:
                            if not trips_can_be_consecutive(self.trips[j], self.trips[i], self.station_order):
                                # Trip i cannot follow trip j due to linear route constraints
                                model.AddBoolOr([
                                    assignment[i][train_idx].Not(),
                                    assignment[j][train_idx].Not()
                                ])
        
        # Constraint 4: Maintenance trains must end at hub stations
        for train_idx, train_name in enumerate(self.train_names):
            if train_name in self.maintenance_trains:
                # Find trips that end at hub stations
                hub_ending_trips = []
                for i, trip in enumerate(self.trips):
                    _, end_stop = get_trip_endpoints(trip)
                    if end_stop in self.hub_stops:
                        hub_ending_trips.append(assignment[i][train_idx])
                
                if hub_ending_trips:
                    # At least one trip ending at hub must be assigned to this train
                    model.AddBoolOr(hub_ending_trips)
        
        # Constraint 5: Load balancing - minimize maximum trips per train
        trips_per_train = []
        for train_idx in range(self.num_trains):
            trips_count = model.NewIntVar(0, len(self.trips), f'trips_count_train_{train_idx}')
            model.Add(trips_count == sum(assignment[i][train_idx] for i in range(len(self.trips))))
            trips_per_train.append(trips_count)
        
        # Minimize maximum trips per train
        max_trips = model.NewIntVar(0, len(self.trips), 'max_trips_per_train')
        for trips_count in trips_per_train:
            model.Add(max_trips >= trips_count)
        
        model.Minimize(max_trips)
        
        # Solve the model
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        solver.parameters.num_search_workers = 8  # Use parallel solving
        solver.parameters.log_search_progress = True
        
        logger.info(f"Starting solve with {time_limit_seconds}s time limit...")
        start_time = time.time()
        
        status = solver.Solve(model)
        
        solve_time = time.time() - start_time
        logger.info(f"Solver finished in {solve_time:.2f}s")
        
        if status == cp_model.OPTIMAL:
            logger.info("Found optimal solution")
        elif status == cp_model.FEASIBLE:
            logger.info("Found feasible solution")
        else:
            logger.error(f"No solution found. Status: {solver.StatusName(status)}")
            return None
        
        # Extract solution and sort trips by start time for each train
        schedule = defaultdict(list)
        for i in range(len(self.trips)):
            for j in range(self.num_trains):
                if solver.Value(assignment[i][j]) == 1:
                    train_name = self.train_names[j]
                    trip_id = self.trips[i]['trip_id']
                    start_time, _ = calculate_trip_duration(self.trips[i])
                    schedule[train_name].append((trip_id, start_time))
                    break
        
        # Sort trips by start time for each train and extract trip IDs
        sorted_schedule = {}
        for train_name, trip_list in schedule.items():
            # Sort by start time (second element of tuple)
            sorted_trips = sorted(trip_list, key=lambda x: x[1])
            # Extract just the trip IDs in chronological order
            sorted_schedule[train_name] = [trip_id for trip_id, _ in sorted_trips]
        
        # Log statistics
        trips_counts = [len(trips) for trips in sorted_schedule.values()]
        logger.info(f"Solution statistics:")
        logger.info(f"  Max trips per train: {max(trips_counts) if trips_counts else 0}")
        logger.info(f"  Min trips per train: {min(trips_counts) if trips_counts else 0}")
        logger.info(f"  Avg trips per train: {sum(trips_counts)/len(trips_counts) if trips_counts else 0:.1f}")
        logger.info(f"  Total trips assigned: {sum(trips_counts)}")
        
        return dict(sorted_schedule)

def save_schedule(schedule: Dict[str, List[str]], filename: str, trips: List[Dict]):
    """
    Save the train schedule to a JSON file with detailed timing information.
    
    Args:
        schedule: Dictionary mapping train names to trip ID lists (already sorted)
        filename: Output filename
        trips: Original trips list for timing information
    """
    # Create trip lookup for timing info
    trip_lookup = {trip['trip_id']: trip for trip in trips}
    
    # Create detailed schedule with timing information
    detailed_schedule = {}
    
    for train, trip_ids in sorted(schedule.items()):
        train_details = []
        
        for i, trip_id in enumerate(trip_ids):
            trip = trip_lookup[trip_id]
            start_time, end_time = calculate_trip_duration(trip)
            start_stop, end_stop = get_trip_endpoints(trip)
            
            trip_detail = {
                "sequence": i + 1,
                "trip_id": trip_id,
                "start_time": seconds_to_time(start_time),
                "end_time": seconds_to_time(end_time),
                "start_station": start_stop,
                "end_station": end_stop,
                "duration_minutes": (end_time - start_time) // 60
            }
            train_details.append(trip_detail)
        
        detailed_schedule[train] = {
            "total_trips": len(trip_ids),
            "first_departure": train_details[0]["start_time"] if train_details else "N/A",
            "last_arrival": train_details[-1]["end_time"] if train_details else "N/A",
            "trips": train_details
        }
    
    # Save detailed schedule
    with open(filename, 'w') as f:
        json.dump(detailed_schedule, f, indent=2, sort_keys=True)
    
    # Also save simple format for backward compatibility
    simple_schedule = {train: trip_ids for train, trip_ids in sorted(schedule.items())}
    simple_filename = filename.replace('.json', '_simple.json')
    with open(simple_filename, 'w') as f:
        json.dump(simple_schedule, f, indent=2, sort_keys=True)
    
    logger.info(f"Detailed schedule saved to {filename}")
    logger.info(f"Simple schedule saved to {simple_filename}")

def validate_schedule(schedule: Dict[str, List[str]], trips: List[Dict],
                     maintenance_trains: Set[str], hub_stops: Set[str]) -> Tuple[bool, Dict]:
    """
    Comprehensive validation of the generated schedule against all constraints.
    
    Args:
        schedule: Generated schedule
        trips: Original trips list
        maintenance_trains: Trains requiring maintenance
        hub_stops: Hub stations for maintenance
    
    Returns:
        Tuple of (is_valid: bool, validation_report: Dict)
    """
    logger.info("Validating schedule...")
    
    # Create trip lookup
    trip_lookup = {trip['trip_id']: trip for trip in trips}
    
    # Initialize validation report
    validation_report = {
        "total_constraints": 0,
        "passed_constraints": 0,
        "failed_constraints": [],
        "warnings": [],
        "constraint_details": {},
        "performance_metrics": {}
    }
    
    def add_constraint_result(name: str, passed: bool, details: str = "", is_critical: bool = True):
        validation_report["total_constraints"] += 1
        validation_report["constraint_details"][name] = {
            "passed": passed,
            "details": details,
            "critical": is_critical
        }
        if passed:
            validation_report["passed_constraints"] += 1
        else:
            if is_critical:
                validation_report["failed_constraints"].append(f"{name}: {details}")
            else:
                validation_report["warnings"].append(f"{name}: {details}")
    
    # Constraint 1: Each trip assigned exactly once
    assigned_trips = set()
    duplicate_assignments = []
    
    for train, trips_list in schedule.items():
        for trip_id in trips_list:
            if trip_id in assigned_trips:
                duplicate_assignments.append(trip_id)
            assigned_trips.add(trip_id)
    
    expected_trips = {trip['trip_id'] for trip in trips}
    missing_trips = expected_trips - assigned_trips
    extra_trips = assigned_trips - expected_trips
    
    constraint1_passed = (len(duplicate_assignments) == 0 and 
                         len(missing_trips) == 0 and 
                         len(extra_trips) == 0)
    
    details = []
    if duplicate_assignments:
        details.append(f"Duplicate assignments: {duplicate_assignments}")
    if missing_trips:
        details.append(f"Missing trips: {list(missing_trips)[:5]}{'...' if len(missing_trips) > 5 else ''}")
    if extra_trips:
        details.append(f"Extra trips: {list(extra_trips)[:5]}{'...' if len(extra_trips) > 5 else ''}")
    
    add_constraint_result("Trip Assignment Uniqueness", constraint1_passed, "; ".join(details))
    
    # Constraint 2: No overlapping trips per train
    overlap_violations = []
    continuity_violations = []
    
    for train, trips_list in schedule.items():
        if len(trips_list) <= 1:
            continue
            
        # Get trip details with timing
        train_trip_details = []
        for trip_id in trips_list:
            if trip_id in trip_lookup:
                trip = trip_lookup[trip_id]
                start_time, end_time = calculate_trip_duration(trip)
                start_stop, end_stop = get_trip_endpoints(trip)
                train_trip_details.append({
                    'trip_id': trip_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_stop': start_stop,
                    'end_stop': end_stop
                })
        
        # Sort by start time
        train_trip_details.sort(key=lambda x: x['start_time'])
        
        # Check for overlaps and continuity
        for i in range(len(train_trip_details) - 1):
            trip1 = train_trip_details[i]
            trip2 = train_trip_details[i + 1]
            
            # Check overlap
            if trip1['end_time'] > trip2['start_time']:
                overlap_violations.append(
                    f"{train}: {trip1['trip_id']} ends at {seconds_to_time(trip1['end_time'])} "
                    f"but {trip2['trip_id']} starts at {seconds_to_time(trip2['start_time'])}"
                )
            
            # Check continuity (can trip2 follow trip1?)
            if not trips_can_be_consecutive(trip_lookup[trip1['trip_id']], 
                                          trip_lookup[trip2['trip_id']], 
                                          STATION_ORDER):
                continuity_violations.append(
                    f"{train}: {trip1['trip_id']} ends at {trip1['end_stop']} "
                    f"but {trip2['trip_id']} starts at {trip2['start_stop']} - insufficient time/impossible route"
                )
    
    add_constraint_result("No Trip Overlaps", len(overlap_violations) == 0, 
                         f"{len(overlap_violations)} violations: {overlap_violations[:3]}")
    
    add_constraint_result("Trip Continuity", len(continuity_violations) == 0,
                         f"{len(continuity_violations)} violations: {continuity_violations[:3]}")
    
    # Constraint 4: Maintenance trains end at hub stations
    maintenance_violations = []
    
    for train, trips_list in schedule.items():
        if train in maintenance_trains and trips_list:
            # Find last trip by end time
            last_trip_id = None
            latest_end_time = -1
            
            for trip_id in trips_list:
                if trip_id in trip_lookup:
                    _, end_time = calculate_trip_duration(trip_lookup[trip_id])
                    if end_time > latest_end_time:
                        latest_end_time = end_time
                        last_trip_id = trip_id
            
            if last_trip_id:
                _, end_stop = get_trip_endpoints(trip_lookup[last_trip_id])
                if end_stop not in hub_stops:
                    maintenance_violations.append(
                        f"{train} ends at {end_stop} (not a hub station)"
                    )
    
    add_constraint_result("Maintenance Hub Constraint", len(maintenance_violations) == 0,
                         "; ".join(maintenance_violations))
    
    # Performance Metrics
    trips_per_train = [len(trips_list) for trips_list in schedule.values()]
    
    if trips_per_train:
        validation_report["performance_metrics"] = {
            "total_trips_assigned": sum(trips_per_train),
            "total_trips_expected": len(trips),
            "assignment_rate": sum(trips_per_train) / len(trips) * 100,
            "max_trips_per_train": max(trips_per_train),
            "min_trips_per_train": min(trips_per_train),
            "avg_trips_per_train": sum(trips_per_train) / len(trips_per_train),
            "std_dev_trips": (sum([(x - sum(trips_per_train)/len(trips_per_train))**2 
                                 for x in trips_per_train]) / len(trips_per_train)) ** 0.5,
            "load_balance_score": 100 - (max(trips_per_train) - min(trips_per_train)) / max(trips_per_train) * 100,
            "trains_utilized": len([t for t in trips_per_train if t > 0]),
            "utilization_rate": len([t for t in trips_per_train if t > 0]) / len(schedule) * 100
        }
    
    # Calculate overall accuracy
    accuracy = validation_report["passed_constraints"] / validation_report["total_constraints"] * 100 if validation_report["total_constraints"] > 0 else 0
    validation_report["overall_accuracy"] = accuracy
    
    # Overall validity (all critical constraints must pass)
    critical_failures = [name for name, details in validation_report["constraint_details"].items() 
                        if details["critical"] and not details["passed"]]
    
    is_valid = len(critical_failures) == 0
    
    logger.info(f"Validation completed - Accuracy: {accuracy:.1f}%, Valid: {is_valid}")
    
    return is_valid, validation_report

def create_sample_trips_file():
    """Create a sample trips.json file for testing with realistic linear route trips."""
    sample_trips = []
    trip_id = 1
    
    # Create trips in both directions along the linear route
    # Some trips from ALVA towards TPHT, some from TPHT towards ALVA
    
    # Morning trips - mostly ALVA to various stations (outbound)
    for i in range(8):
        start_hour = 6 + i // 2
        start_minute = (i % 2) * 30
        
        # Pick stations along the route
        start_idx = 0  # ALVA
        end_idx = 5 + (i * 3) % 15  # Various stations towards TPHT
        
        trip = {
            "trip_id": f"TRIP_{trip_id:03d}",
            "stops": [
                {
                    "stop_id": STATION_ORDER[start_idx],
                    "arrival_time": f"{start_hour:02d}:{start_minute:02d}:00",
                    "departure_time": f"{start_hour:02d}:{start_minute + 2:02d}:00"
                },
                {
                    "stop_id": STATION_ORDER[end_idx],
                    "arrival_time": f"{start_hour:02d}:{start_minute + 20 + end_idx * 2:02d}:00",
                    "departure_time": f"{start_hour:02d}:{start_minute + 22 + end_idx * 2:02d}:00"
                }
            ]
        }
        sample_trips.append(trip)
        trip_id += 1
    
    # Evening trips - mostly towards ALVA (return journeys)
    for i in range(8):
        start_hour = 16 + i // 2
        start_minute = (i % 2) * 30
        
        # Pick stations along the route (reverse direction)
        start_idx = 15 + (i * 2) % 10  # Various stations
        end_idx = max(0, start_idx - 8 - i)  # Towards ALVA
        
        trip = {
            "trip_id": f"TRIP_{trip_id:03d}",
            "stops": [
                {
                    "stop_id": STATION_ORDER[start_idx],
                    "arrival_time": f"{start_hour:02d}:{start_minute:02d}:00",
                    "departure_time": f"{start_hour:02d}:{start_minute + 2:02d}:00"
                },
                {
                    "stop_id": STATION_ORDER[end_idx],
                    "arrival_time": f"{start_hour:02d}:{start_minute + 20 + abs(start_idx - end_idx) * 2:02d}:00",
                    "departure_time": f"{start_hour:02d}:{start_minute + 22 + abs(start_idx - end_idx) * 2:02d}:00"
                }
            ]
        }
        sample_trips.append(trip)
        trip_id += 1
    
    # Add some full-length trips (ALVA to TPHT and vice versa)
    for i in range(4):
        start_hour = 8 + i * 4
        
        # Alternating direction full trips
        if i % 2 == 0:
            # ALVA to TPHT
            start_stop = STATION_ORDER[0]
            end_stop = STATION_ORDER[-1]
            travel_time = len(STATION_ORDER) * 2  # 2 minutes per station
        else:
            # TPHT to ALVA  
            start_stop = STATION_ORDER[-1]
            end_stop = STATION_ORDER[0]
            travel_time = len(STATION_ORDER) * 2
        
        trip = {
            "trip_id": f"TRIP_{trip_id:03d}",
            "stops": [
                {
                    "stop_id": start_stop,
                    "arrival_time": f"{start_hour:02d}:00:00",
                    "departure_time": f"{start_hour:02d}:02:00"
                },
                {
                    "stop_id": end_stop,
                    "arrival_time": f"{start_hour:02d}:{travel_time + 10:02d}:00",
                    "departure_time": f"{start_hour:02d}:{travel_time + 12:02d}:00"
                }
            ]
        }
        sample_trips.append(trip)
        trip_id += 1
    
    with open('trips.json', 'w') as f:
        json.dump(sample_trips, f, indent=2)
    
    logger.info(f"Created sample trips.json file with {len(sample_trips)} trips")
    logger.info("Trips include realistic patterns for a linear ALVA-TPHT route")

def main():
    """Main function to run the train scheduler."""
    try:
        # Check if trips file exists, create sample if not
        try:
            trips = load_trips('trips.json')
        except FileNotFoundError:
            logger.info("trips.json not found, creating sample file...")
            create_sample_trips_file()
            trips = load_trips('trips.json')
        
        # Initialize and run scheduler
        scheduler = TrainScheduler(
            trips=trips,
            num_trains=NUM_TRAINS,
            maintenance_trains=MAINTENANCE_TRAINS,
            hub_stops=HUB_STOPS,
            station_order=STATION_ORDER
        )
        
        # Solve with 2-minute time limit
        schedule = scheduler.solve(time_limit_seconds=120)
        
        if schedule:
            # Validate solution with comprehensive analysis
            is_valid, validation_report = validate_schedule(schedule, trips, MAINTENANCE_TRAINS, HUB_STOPS)
            
            # Calculate quality scores
            quality_scores = calculate_schedule_quality_score(validation_report)
            
            # Print detailed validation report
            print_validation_report(validation_report)
            
            # Print quality assessment
            print(f"\n🏆 QUALITY ASSESSMENT:")
            print("-" * 50)
            print(f"Overall Quality Score: {quality_scores['overall_quality']:.1f}/100 (Grade: {quality_scores['grade']})")
            print(f"Constraint Compliance: {quality_scores['constraint_compliance']:.1f}%")
            print(f"Load Balance:          {quality_scores['load_balance']:.1f}%") 
            print(f"Resource Utilization:  {quality_scores['utilization']:.1f}%")
            print(f"Assignment Complete:   {quality_scores['assignment_completeness']:.1f}%")
            print(f"Efficiency:           {quality_scores['efficiency']:.1f}%")
            
            # Save results
            save_schedule(schedule, 'train_schedule.json', trips)
            
            # Save validation report
            with open('validation_report.json', 'w') as f:
                validation_data = {
                    "validation_report": validation_report,
                    "quality_scores": quality_scores,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                json.dump(validation_data, f, indent=2)
            logger.info("Validation report saved to validation_report.json")
            
            if is_valid:
                logger.info("✅ Schedule generation completed successfully!")
                
                # Print detailed summary only if valid
                print("\n" + "="*60)
                print("TRAIN SCHEDULE SUMMARY (Chronological Order)")
                print("="*60)
                
                # Create trip lookup for timing
                trip_lookup = {trip['trip_id']: trip for trip in trips}
                
                for train in sorted(schedule.keys()):
                    trip_ids = schedule[train]
                    if not trip_ids:
                        continue
                        
                    print(f"\n🚆 {train} ({len(trip_ids)} trips):")
                    print("-" * 40)
                    
                    for i, trip_id in enumerate(trip_ids):
                        trip = trip_lookup[trip_id]
                        start_time, end_time = calculate_trip_duration(trip)
                        start_stop, end_stop = get_trip_endpoints(trip)
                        
                        print(f"  {i+1:2d}. {trip_id} | {seconds_to_time(start_time)} → {seconds_to_time(end_time)} | {start_stop} → {end_stop}")
                    
                    # Show first and last trip summary
                    first_trip = trip_lookup[trip_ids[0]]
                    last_trip = trip_lookup[trip_ids[-1]]
                    first_start, _ = calculate_trip_duration(first_trip)
                    _, last_end = calculate_trip_duration(last_trip)
                    
                    print(f"     📋 Operating: {seconds_to_time(first_start)} to {seconds_to_time(last_end)}")
                    
                    # Check if it's a maintenance train
                    if train in MAINTENANCE_TRAINS:
                        _, last_stop = get_trip_endpoints(last_trip)
                        if last_stop in HUB_STOPS:
                            print(f"     🔧 Maintenance: Ends at hub {last_stop} ✓")
                        else:
                            print(f"     ⚠️  Maintenance: Ends at {last_stop} (NOT a hub!)")
                
                print("\n" + "="*60)
            else:
                logger.error("❌ Schedule has constraint violations!")
        else:
            logger.error("❌ No feasible schedule found")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
