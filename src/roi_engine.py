import pandas as pd

class ROIEngine:
    def __init__(self, cost_per_failure=50000, cost_per_maintenance=2000, cost_of_sensor_deployment=10000):
        """
        cost_per_failure: USD lost per unplanned downtime event.
        cost_per_maintenance: USD spent on a scheduled maintenance task.
        cost_of_sensor_deployment: Initial investment in the monitoring system.
        """
        self.cost_per_failure = cost_per_failure
        self.cost_per_maintenance = cost_per_maintenance
        self.cost_of_sensor_deployment = cost_of_sensor_deployment

    def calculate_impact(self, predictions_df):
        """
        Calculates the financial impact of moving from reactive to predictive maintenance.
        Assumes we perform maintenance when predicted RUL < 48 hours.
        """
        # Reactive approach: We only fix it AFTER failure (RUL reaches 0)
        # In this dataset, let's assume failures are captured by our data generator.
        # For simplicity in this simulation, we look at the number of "preventable" maintenances.
        
        total_events = len(predictions_df)
        
        # Preventive: We intervene when RUL < 48 hours
        preventive_maintenance_count = len(predictions_df[predictions_df['predicted_RUL'] <= 48])
        
        # Cost of Reactive Approach (Cost of failures we didn't prevent)
        # For the sake of this simulation, assume every failure event costs cost_per_failure.
        # A 'failure' in our data is when RUL reaches 0.
        actual_failures = len(predictions_df[predictions_df['RUL_hours'] <= 0])
        reactive_cost = actual_failures * self.cost_per_failure
        
        # Cost of Predictive Approach:
        # We pay for the preventative maintenance tasks
        predictive_maintenance_cost = preventive_maintenance_count * self.cost_per_maintenance
        
        # Total cost is (Maintenance costs) + (Initial setup)
        total_predictive_cost = predictive_maintenance_score = (preventive_maintenance_count * self.cost_per_maintenance) + self.cost_of_sensor_deployment
        
        roi_savings = reactive_cost - total_predictive_cost
        roi_percent = (roi_savings / reactive_cost) * 100 if reactive_cost > 0 else 0

        return {
            "reactive_total_cost": reactive_cost,
            "predictive_total_cost": total_predictive_cost,
            "net_savings": roi_savings,
            "roi_percentage": roi_percent
        }

if __name__ == "__main__":
    # Test with dummy data
    df = pd.DataFrame({'predicted_RUL': [10, 5, 100, 2, 30], 'RUL_hours': [0, 0, 80, 0, 40]})
    engine = ROIEngine()
    results = engine.calculate_impact(df)
    print("ROI Test Results:", results)
