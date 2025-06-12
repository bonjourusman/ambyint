import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class WellOptimizer:
    """
    Oil Well Optimization System
    """
    
    def __init__(self):
        self.production_model = None
        self.baseline_metrics = {}
        self.health_thresholds = {}
        self.weekly_reports = []
        
    def prepare_features(self, df):
        """
        Create essential features for analysis
        """
        df = df.copy()
        
        # Core engineered features
        df['pressure_differential'] = df['Casing_Pressure_psi'] - df['Tubing_Pressure_psi']
        df['efficiency_ratio'] = df['Liquid_Flow_Rate_m3d'] / (df['Motor_Power_kW'] + 1e-6)
        df['thermal_stress'] = df['Gearbox_Temperature_C'] - df['Ambient_Temperature_C']
        df['vibration_composite'] = (df['Gearbox_Vibration_mm_s'] + df['Polished_Rod_Vibration_mm_s']) / 2
        
        # Fill missing values in quantitative columns
        quant_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[quant_cols] = df[quant_cols].fillna(df[quant_cols].rolling(window=7, min_periods=1, center=True).mean())
        
        return df
    
    def calc_health_thresholds(self, df):
        """
        Calculate equipment health thresholds from high-performance periods
        """
        high_perf_data = df[df['Liquid_Flow_Rate_m3d'] >= df['Liquid_Flow_Rate_m3d'].quantile(0.90)]
        
        self.health_thresholds = {
            'vibration': {
                'excellent_min': high_perf_data['vibration_composite'].quantile(0.25),
                'excellent_max': high_perf_data['vibration_composite'].quantile(0.75)
            },
            'temperature': {
                'excellent_min': high_perf_data['Gearbox_Temperature_C'].quantile(0.25),
                'excellent_max': high_perf_data['Gearbox_Temperature_C'].quantile(0.75)
            }
        }
        return self.health_thresholds
    
    def calc_equip_health(self, df):
        """
        Calculate equipment health score
        """
        df = df.copy()
        
        def health_score(value, thresholds, penalty_rate):
            if thresholds['excellent_min'] <= value <= thresholds['excellent_max']:
                return 100
            distance = min(abs(value - thresholds['excellent_min']), abs(value - thresholds['excellent_max']))
            return max(0, 100 - (distance * penalty_rate))
        
        # Optimal ranges for Vibration and Temperature
        vib_range = self.health_thresholds['vibration']['excellent_max'] - self.health_thresholds['vibration']['excellent_min']
        temp_range = self.health_thresholds['temperature']['excellent_max'] - self.health_thresholds['temperature']['excellent_min']

        # Penalty rates for Vibration and Temperature variations
        vib_penalty_rate = 100 / (5 * vib_range)
        temp_penalty_rate = 100 / (5 * temp_range)
        
        # Calculate health scores for Vibration and Temperature
        vib_health = df['vibration_composite'].apply(lambda x: health_score(x, self.health_thresholds['vibration'], vib_penalty_rate))
        temp_health = df['Gearbox_Temperature_C'].apply(lambda x: health_score(x, self.health_thresholds['temperature'], temp_penalty_rate))
        
        # Pressure stability
        high_perf_data = df[df['Liquid_Flow_Rate_m3d'] >= df['Liquid_Flow_Rate_m3d'].quantile(0.90)]
        pressure_std = high_perf_data['pressure_differential'].std()
        pressure_penalty_rate = 100 / (5 * pressure_std)
        pressure_mean = df['pressure_differential'].mean()
        pressure_deviation = np.abs(df['pressure_differential'] - pressure_mean)
        pressure_health = np.clip(100 - (pressure_deviation * pressure_penalty_rate), 0, 100)
        
        # Combined health score
        df['Equipment_Health_Score'] = (vib_health * 0.4 + temp_health * 0.4 + pressure_health * 0.2)
        
        return df
    
    def train_model(self, training_data):
        """
        Train production prediction model
        """
        print("Training production prediction model...")
        
        train_df = self.prepare_features(training_data)
        self.calc_health_thresholds(train_df)
        train_df = self.calc_equip_health(train_df)
        
        # Calculate baselines
        is_high_perf = train_df['Liquid_Flow_Rate_m3d'] > train_df['Liquid_Flow_Rate_m3d'].quantile(0.99)
        self.baseline_metrics = {
            'production_avg': train_df['Liquid_Flow_Rate_m3d'].mean(),
            'production_std': train_df['Liquid_Flow_Rate_m3d'].std(),
            'health_avg': train_df['Equipment_Health_Score'].mean(),
            'health_std': train_df['Equipment_Health_Score'].std(),
            'efficiency_avg': train_df['efficiency_ratio'].mean(),
            'efficiency_std': train_df['efficiency_ratio'].std(),
            'optimal_vfd_range': [
                train_df.loc[is_high_perf, 'VFD_Speed_pct'].min(),
                train_df.loc[is_high_perf, 'VFD_Speed_pct'].max()
            ],
            'baseline_vfd_std': train_df['VFD_Speed_pct'].std(),
            'baseline_power_std': train_df['Motor_Power_kW'].std()
        }

        # Train production model
        features = ['VFD_Speed_pct', 'Motor_Power_kW', 'Stroke_Count_Today', 'Hydraulic_Pressure_psi', 'thermal_stress', 'pressure_differential', 'efficiency_ratio', 'vibration_composite']
        X = train_df[features]
        y = train_df['Liquid_Flow_Rate_m3d']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.production_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.production_model.fit(X_train, y_train)
        
        # Model performance
        y_pred = self.production_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\nModel trained - R²: {r2:.3f}, RMSE: {rmse:.2f} m³/d")
        print(f"Baseline Production: {self.baseline_metrics['production_avg']:.2f} m³/d")
        
        return train_df
    
    def get_thresholds(self):
        """
        Get all thresholds derived from training data
        """
        return {
            'production_concerning': -(self.baseline_metrics['production_std'] / self.baseline_metrics['production_avg']) * 100,
            'production_critical': -2 * (self.baseline_metrics['production_std'] / self.baseline_metrics['production_avg']) * 100,
            'health_concerning': self.baseline_metrics['health_avg'] - (0.5 * self.baseline_metrics['health_std']),
            'health_critical': self.baseline_metrics['health_avg'] - (1 * self.baseline_metrics['health_std']),
            'vfd_deviation': self.baseline_metrics['baseline_vfd_std'],
            'vfd_critical_deviation': self.baseline_metrics['baseline_vfd_std'] * 2,
            'power_deviation': self.baseline_metrics['baseline_power_std'],
            'meaningful_improvement': (self.baseline_metrics['production_std'] / self.baseline_metrics['production_avg']) * 50
        }
    
    def generate_recommendations(self, weekly_df, metrics):
        """
        Generate specific recommendations
        """
        recommendations = []
        thresholds = self.get_thresholds()
        
        #print('Production vs Baseline Metric: ', metrics['production_vs_baseline'])
        #print('Production Critical Threshold: ', thresholds['production_critical'])

        # Production alerts
        if metrics['production_vs_baseline'] < thresholds['production_critical']:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Production',
                'action': f"Production {metrics['production_vs_baseline']:.1f}% below baseline - Check VFD and motor settings"
            })
        elif metrics['production_vs_baseline'] < thresholds['production_concerning']:
            recommendations.append({
                'priority': 'MEDIUM', 
                'category': 'Production',
                'action': f"Production {metrics['production_vs_baseline']:.1f}% below baseline - Consider VFD adjustment"
            })
        
        # Health alerts
        if metrics['health_avg'] < thresholds['health_critical']:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Equipment Health',
                'action': f"Equipment health: {metrics['health_avg']:.1f}/100 - Schedule inspection"
            })
        elif metrics['health_avg'] < thresholds['health_concerning']:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Equipment Health',
                'action': f"Equipment health: {metrics['health_avg']:.1f}/100 - Monitor closely"
            })
        
        # VFD optimization
        optimal_vfd_range = self.baseline_metrics['optimal_vfd_range']
        current_vfd = metrics['avg_vfd_speed']
        
        if current_vfd < optimal_vfd_range[0] - thresholds['vfd_critical_deviation']:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'VFD Critical',
                'action': f"VFD critically low: {current_vfd:.1f}% - Urgently increase to {optimal_vfd_range[0]:.0f}%"
            })
        elif current_vfd < optimal_vfd_range[0] - thresholds['vfd_deviation']:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'VFD Optimization', 
                'action': f"Increase VFD speed from {current_vfd:.1f}% to {optimal_vfd_range[0]:.0f}%"
            })
        elif current_vfd > optimal_vfd_range[1] + thresholds['vfd_critical_deviation']:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'VFD Critical',
                'action': f"VFD critically high: {current_vfd:.1f}% - Urgently reduce to {optimal_vfd_range[1]:.0f}%"
            })
        elif current_vfd > optimal_vfd_range[1] + thresholds['vfd_deviation']:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'VFD Optimization',
                'action': f"Reduce VFD speed from {current_vfd:.1f}% to {optimal_vfd_range[1]:.0f}%"
            })
        
        return recommendations
    
    def generate_scenarios(self, weekly_df):
        """
        Generate what-if scenarios for VFD as its an operator-adjustable parameter to optimize production.
        """
        current_vfd = weekly_df['VFD_Speed_pct'].mean()
        
        thresholds = self.get_thresholds()
        optimal_vfd_range = self.baseline_metrics['optimal_vfd_range']
        optimal_vfd = np.mean(optimal_vfd_range)

        scenarios = []
        print(f"  VFD Analysis: Current={current_vfd:.1f}%, Optimal Range={optimal_vfd_range[0]:.1f}%-{optimal_vfd_range[1]:.1f}%, Center={optimal_vfd:.1f}%")
        
        # Check if VFD is critically low
        if current_vfd < optimal_vfd_range[0] - thresholds['vfd_critical_deviation']:
            scenarios.append({
                'name': 'VFD Critical Low Correction',
                'changes': {'VFD_Speed_pct': optimal_vfd_range[0]},
                'description': f'CRITICAL: Increase VFD from {current_vfd:.1f}% to {optimal_vfd_range[0]:.1f}% (minimum safe level)'
            })
            
        # Check if VFD is critically high  
        elif current_vfd > optimal_vfd_range[1] + thresholds['vfd_critical_deviation']:
            scenarios.append({
                'name': 'VFD Critical High Correction',
                'changes': {'VFD_Speed_pct': optimal_vfd_range[1]},
                'description': f'CRITICAL: Reduce VFD from {current_vfd:.1f}% to {optimal_vfd_range[1]:.1f}% (maximum safe level)'
            })
            
        # Check if VFD is moderately low (non-critical)
        elif current_vfd < optimal_vfd_range[0] - thresholds['vfd_deviation']:
            # Offer both conservative and optimal adjustment options
            scenarios.append({
                'name': 'VFD Moderate Increase',
                'changes': {'VFD_Speed_pct': optimal_vfd_range[0]},
                'description': f'Increase VFD from {current_vfd:.1f}% to {optimal_vfd_range[0]:.1f}% (optimal minimum)'
            })
            scenarios.append({
                'name': 'VFD Optimal Adjustment',
                'changes': {'VFD_Speed_pct': optimal_vfd},
                'description': f'Increase VFD from {current_vfd:.1f}% to {optimal_vfd:.1f}% (optimal center)'
            })
            
        # Check if VFD is moderately high (non-critical)
        elif current_vfd > optimal_vfd_range[1] + thresholds['vfd_deviation']:
            # Offer both conservative and optimal adjustment options
            scenarios.append({
                'name': 'VFD Moderate Decrease',
                'changes': {'VFD_Speed_pct': optimal_vfd_range[1]},
                'description': f'Reduce VFD from {current_vfd:.1f}% to {optimal_vfd_range[1]:.1f}% (optimal maximum)'
            })
            scenarios.append({
                'name': 'VFD Optimal Adjustment',
                'changes': {'VFD_Speed_pct': optimal_vfd},
                'description': f'Reduce VFD from {current_vfd:.1f}% to {optimal_vfd:.1f}% (optimal center)'
            })

        # Check if VFD is outside optimal range (below)
        elif current_vfd < optimal_vfd_range[0]:
            scenarios.append({
                'name': 'VFD Range Correction (Low)',
                'changes': {'VFD_Speed_pct': optimal_vfd_range[0]},
                'description': f'Bring VFD into optimal range: {current_vfd:.1f}% → {optimal_vfd_range[0]:.1f}%'
            })
            scenarios.append({
                'name': 'VFD Optimal Center',
                'changes': {'VFD_Speed_pct': optimal_vfd},
                'description': f'Move VFD to optimal center: {current_vfd:.1f}% → {optimal_vfd:.1f}%'
            })
            
        # Check if VFD is outside optimal range (above)
        elif current_vfd > optimal_vfd_range[1]:
            scenarios.append({
                'name': 'VFD Range Correction (High)',
                'changes': {'VFD_Speed_pct': optimal_vfd_range[1]},
                'description': f'Bring VFD into optimal range: {current_vfd:.1f}% → {optimal_vfd_range[1]:.1f}%'
            })
            scenarios.append({
                'name': 'VFD Optimal Center',
                'changes': {'VFD_Speed_pct': optimal_vfd},
                'description': f'Move VFD to optimal center: {current_vfd:.1f}% → {optimal_vfd:.1f}%'
            })
        
        # Always test some VFD adjustments for comparison
        else:
            # VFD is within optimal range - test small adjustments
            scenarios.append({
                'name': 'VFD Increase Test',
                'changes': {'VFD_Speed_pct': min(current_vfd + 5, optimal_vfd_range[1])},
                'description': f'Test VFD increase: {current_vfd:.1f}% → {min(current_vfd + 5, optimal_vfd_range[1]):.1f}%'
            })
            scenarios.append({
                'name': 'VFD Decrease Test', 
                'changes': {'VFD_Speed_pct': max(current_vfd - 5, optimal_vfd_range[0])},
                'description': f'Test VFD decrease: {current_vfd:.1f}% → {max(current_vfd - 5, optimal_vfd_range[0]):.1f}%'
            })
            
            # Only add center optimization if significantly different
            if abs(current_vfd - optimal_vfd) > 0.5:
                scenarios.append({
                    'name': 'VFD Center Optimization',
                    'changes': {'VFD_Speed_pct': optimal_vfd},
                    'description': f'Optimize to center: {current_vfd:.1f}% → {optimal_vfd:.1f}%'
                })
        
        # Predict scenario outcomes
        features = ['VFD_Speed_pct', 'Motor_Power_kW', 'Stroke_Count_Today', 'Hydraulic_Pressure_psi', 
                   'thermal_stress', 'pressure_differential', 'efficiency_ratio', 'vibration_composite']
        
        scenario_results = []
        
        for scenario in scenarios:
            modified_df = weekly_df.copy()
            for param, value in scenario['changes'].items():
                if param in modified_df.columns:
                    modified_df[param] = value
            
            modified_df = self.prepare_features(modified_df)
            X_scenario = modified_df[features]
            predicted_production = self.production_model.predict(X_scenario).mean()
            current_production = weekly_df['Liquid_Flow_Rate_m3d'].mean()

            #print('Predicted Production: ', predicted_production)
            #print('Current Production: ', current_production)

            improvement = ((predicted_production - current_production) / current_production) * 100
            
            #print('Improvement: ', improvement)
            #print('Threshold Meaningful Improvement: ', thresholds['meaningful_improvement'])
            print(f"  {scenario['name']}: {improvement:+.2f}% improvement")

            # Recommendation logic
            scenario_name = scenario['name']
            if 'Critical' in scenario_name:
                # Critical scenarios always get IMPLEMENTED regardless of improvement
                recommendation = 'IMPLEMENT'
            elif 'Test' in scenario_name:
                # Test scenarios get special treatment
                if improvement > 0.1:  # Very low threshold for tests
                    recommendation = 'MONITOR'
                else:
                    recommendation = 'AVOID'
            elif improvement > thresholds['meaningful_improvement']:
                recommendation = 'IMPLEMENT'
            elif improvement > 0.1:  # Lower threshold than before
                recommendation = 'MONITOR'
            else:
                recommendation = 'AVOID'
            
            
            scenario_results.append({
                'scenario': scenario['name'],
                'description': scenario['description'],
                'changes': scenario['changes'],
                'improvement_pct': improvement,
                'recommendation': recommendation
            })
        
        return scenario_results
    
    def analyze_week(self, week_data, week_number):
        """
        Analyze a given week and generate recommendations
        """
        weekly_df = self.prepare_features(week_data)
        weekly_df = self.calc_equip_health(weekly_df)
        
        # Calculate metrics
        metrics = {
            'week_number': week_number,
            'production_avg': weekly_df['Liquid_Flow_Rate_m3d'].mean(),
            'production_vs_baseline': ((weekly_df['Liquid_Flow_Rate_m3d'].mean() - self.baseline_metrics['production_avg']) / self.baseline_metrics['production_avg'] * 100),
            'health_avg': weekly_df['Equipment_Health_Score'].mean(),
            'avg_vfd_speed': weekly_df['VFD_Speed_pct'].mean(),
            'avg_motor_power': weekly_df['Motor_Power_kW'].mean()
        }
        
        # Generate recommendations and scenarios
        recommendations = self.generate_recommendations(weekly_df, metrics)
        scenarios = self.generate_scenarios(weekly_df)
        
        weekly_report = {
            'metrics': metrics,
            'recommendations': recommendations,
            'scenarios': scenarios,
            'data': weekly_df
        }
        
        self.weekly_reports.append(weekly_report)
        return weekly_report
    
    def simulate_optimization_period(self, optimization_data, start_date, end_date):
        """
        Run weekly optimization simulation
        """
        print(f"\nWeekly Optimization Simulation: {start_date} to {end_date}")
        print("=" * 60)
        
        # Filter data
        if 'Timestamp' in optimization_data.columns:
            optimization_data['Timestamp'] = pd.to_datetime(optimization_data['Timestamp'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            period_data = optimization_data[
                (optimization_data['Timestamp'] >= start_dt) & 
                (optimization_data['Timestamp'] <= end_dt)
            ].copy().sort_values('Timestamp')
        else:
            period_data = optimization_data.copy()
        
        # Split into 8 weeks
        period_data['week_number'] = pd.cut(range(len(period_data)), bins=8, labels=['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8'])
        
        for week_num in ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6', 'Week 7', 'Week 8']:
            week_data = period_data[period_data['week_number'] == week_num].copy()
            if len(week_data) == 0:
                continue
            
            print(f"\n{week_num} Analysis ({len(week_data)} records)")
            
            week_number = int(week_num.split()[1])
            weekly_report = self.analyze_week(week_data, week_number)
            
            # Print summary
            metrics = weekly_report['metrics']
            print(f"Production: {metrics['production_avg']:.2f} m³/d ({metrics['production_vs_baseline']:+.1f}% vs baseline)")
            print(f"Equipment Health: {metrics['health_avg']:.1f}/100")
            
            # Print top recommendations
            if weekly_report['recommendations']:
                print(f"Recommendations:")
                for rec in weekly_report['recommendations'][:2]:
                    print(f" [{rec['priority']}] {rec['action']}")
            
            # Print best scenario
            if weekly_report['scenarios']:
                best_scenario = max(weekly_report['scenarios'], key=lambda x: x['improvement_pct'])
                print(f" Best Opportunity: {best_scenario['scenario']}")
                print(f" Expected Gain: +{best_scenario['improvement_pct']:.1f}%")
                print(f" Recommendation: {best_scenario['recommendation']}")
            else:
                # Debug info when no scenarios are generated
                current_vfd = metrics['avg_vfd_speed']
                optimal_range = self.baseline_metrics['optimal_vfd_range']
                optimal_center = np.mean(optimal_range)
                vfd_deviation = week_data['VFD_Speed_pct'].std()
                
                print(f"  No scenarios generated:")
                print(f"   Current VFD: {current_vfd:.1f}% | Optimal Range: {optimal_range[0]:.1f}%-{optimal_range[1]:.1f}% | Center: {optimal_center:.1f}%")
                print(f"   VFD Deviation Threshold: {vfd_deviation:.1f}% | Distance from center: {abs(current_vfd - optimal_center):.1f}%")
        
        return self.weekly_reports
    
    def run_comparative_analysis(self):
        """
        Compare actual vs optimized production
        """
        if not self.weekly_reports:
            return None
        
        print(f"\nCOMPARATIVE ANALYSIS")
        print("=" * 50)
        
        total_actual = 0
        total_optimized = 0
        
        for week_num, report in enumerate(self.weekly_reports, 1):
            week_data = report['data']
            actual_production = week_data['Liquid_Flow_Rate_m3d'].sum()
            
            # Find best implementable scenario
            implementable = [s for s in report['scenarios'] if s['recommendation'] == 'IMPLEMENT']
            
            if implementable:
                best_scenario = max(implementable, key=lambda x: x['improvement_pct'])
                improvement_factor = 1 + (best_scenario['improvement_pct'] / 100)
                optimized_production = actual_production * improvement_factor
                scenario_name = best_scenario['scenario']
            else:
                optimized_production = actual_production
                scenario_name = "No optimization needed"
            
            gain = optimized_production - actual_production
            gain_pct = (gain / actual_production) * 100
            
            print(f"Week {week_num}: {actual_production:,.1f} → {optimized_production:,.1f} m³ ({gain_pct:+.1f}%)")
            print(f"  Applied: {scenario_name}")
            
            total_actual += actual_production
            total_optimized += optimized_production
        
        total_gain = total_optimized - total_actual
        total_gain_pct = (total_gain / total_actual) * 100
        
        print(f"\nPERIOD TOTALS:")
        print(f"Actual: {total_actual:,.1f} m³")
        print(f"Optimized: {total_optimized:,.1f} m³")
        print(f"Total Gain: {total_gain:+,.1f} m³ ({total_gain_pct:+.1f}%)")
        
        return {
            'actual_total': total_actual,
            'optimized_total': total_optimized,
            'total_gain': total_gain,
            'improvement_pct': total_gain_pct
        }

if __name__ == "__main__":
    
    optimizer = WellOptimizer()
    
    # Load data
    df = pd.read_csv("dataset/train.csv", parse_dates=['Timestamp'])
    
    # Split data
    training_data = df[df['Timestamp'] <= '2023-08-22 23:59:59']
    optimization_data = df[
        (df['Timestamp'] >= '2023-08-23 00:00:00') & 
        (df['Timestamp'] <= '2023-10-19 23:59:59')
    ]
    
    # Run analysis
    optimizer.train_model(training_data)
    optimizer.simulate_optimization_period(optimization_data, '2023-08-23', '2023-10-19')
    optimizer.run_comparative_analysis()