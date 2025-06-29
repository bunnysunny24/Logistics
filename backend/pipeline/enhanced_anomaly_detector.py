import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import re
from dataclasses import dataclass

@dataclass
class AnomalyResult:
    """Structured anomaly detection result"""
    id: str
    document_id: str
    anomaly_type: str
    risk_score: float
    severity: str
    description: str
    evidence: List[str]
    recommendations: List[str]
    timestamp: float
    metadata: Dict[str, Any]

class EnhancedAnomalyDetector:
    """Enhanced anomaly detector with ML-like scoring and better integration with RAG"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.anomalies_dir = f"{data_dir}/anomalies"
        os.makedirs(self.anomalies_dir, exist_ok=True)
        
        # Load configuration and historical data
        self.load_configuration()
        self.load_historical_data()
        
        # Initialize scoring models
        self.initialize_scoring_models()
        
        # Load existing anomalies
        self.anomalies = self.load_existing_anomalies()
        
    def load_configuration(self):
        """Load enhanced configuration with domain knowledge"""
        self.config = {
            # Invoice compliance rules
            "invoice_rules": {
                "standard_payment_terms": 30,
                "early_payment_discount_days": 10,
                "late_payment_penalty_rate": 0.05,
                "high_risk_amount_threshold": 10000,
                "duplicate_check_window_days": 30,
                "amount_variance_threshold": 0.20,
                "weekend_processing_risk": 0.7,
                "round_amount_suspicion_threshold": 0.8,
                "payment_terms_tolerance": 7  # Allow 7 days deviation in payment terms
            },
            
            # Shipment anomaly thresholds
            "shipment_rules": {
                "route_deviation_km": 200,
                "delivery_delay_days": 2,
                "value_variance_percentage": 0.25,
                "carrier_change_risk_score": 0.7,
                "customs_delay_threshold_days": 5,
                "route_efficiency_threshold": 0.15,
                "weight_discrepancy_threshold": 0.1
            },
            
            # Fraud detection patterns
            "fraud_patterns": {
                "invoice": {
                    "round_amounts": [100, 500, 1000, 5000, 10000],
                    "suspicious_times": ["weekend", "after_hours", "holiday"],
                    "duplicate_patterns": ["same_amount", "same_date", "same_supplier"],
                    "missing_fields": ["po_number", "approval", "receipt_date"]
                },
                "shipment": {
                    "route_flags": ["unusual_carrier", "route_deviation", "timing_anomaly"],
                    "value_flags": ["high_value", "value_mismatch", "currency_inconsistency"],
                    "document_flags": ["missing_docs", "altered_docs", "fake_tracking"]
                }
            },
            
            # Approved entities
            "approved_entities": {
                "carriers": {
                    "tier_1": ["Global Shipping Inc", "Express Worldwide", "Premium Logistics"],
                    "tier_2": ["Regional Express", "Standard Shipping", "Economy Freight"],
                    "blacklisted": ["Suspicious Cargo", "Fake Logistics"]
                },
                "suppliers": {
                    "ABC Electronics": {"payment_terms": "NET30", "risk_level": "low"},
                    "Tech Solutions Inc": {"payment_terms": "NET30", "risk_level": "low"}, 
                    "Global Logistics": {"payment_terms": "NET45", "risk_level": "low"},
                    "New Supplier Co": {"payment_terms": "NET30", "risk_level": "medium"},
                    "Overseas Parts Ltd": {"payment_terms": "NET60", "risk_level": "medium"},
                    "Problem Supplier Inc": {"payment_terms": "NET15", "risk_level": "high"},
                    "Acme Corp": {"payment_terms": "NET30", "risk_level": "low"},
                    "Beta Industries": {"payment_terms": "NET45", "risk_level": "low"},
                    "Delta Corp": {"payment_terms": "NET30", "risk_level": "low"},
                    "Epsilon Ltd": {"payment_terms": "NET30", "risk_level": "low"}
                }
            }
        }
    
    def load_historical_data(self):
        """Load and analyze historical data for better anomaly detection"""
        self.historical_data = {
            "invoice_patterns": {},
            "shipment_patterns": {},
            "seasonal_adjustments": {},
            "supplier_profiles": {},
            "route_profiles": {}
        }
        
        # Load invoice data
        self._analyze_historical_invoices()
        
        # Load shipment data
        self._analyze_historical_shipments()
        
    def _analyze_historical_invoices(self):
        """Analyze historical invoice data to establish baselines"""
        try:
            # Load comprehensive invoice data
            invoice_file = f"{self.data_dir}/invoices/comprehensive_invoices.csv"
            if os.path.exists(invoice_file):
                df = pd.read_csv(invoice_file)
                
                # Analyze by supplier
                for supplier in df['supplier'].unique():
                    supplier_data = df[df['supplier'] == supplier]
                    
                    self.historical_data["supplier_profiles"][supplier] = {
                        "avg_amount": supplier_data['amount'].mean(),
                        "std_amount": supplier_data['amount'].std(),
                        "typical_terms": supplier_data['payment_terms'].mode().iloc[0] if len(supplier_data) > 0 else "NET30",
                        "avg_discount": supplier_data['early_discount'].mean(),
                        "invoice_count": len(supplier_data),
                        "flagged_rate": (supplier_data['status'] == 'flagged').mean()
                    }
                
                logger.info(f"Analyzed {len(df)} historical invoices for {len(df['supplier'].unique())} suppliers")
                
        except Exception as e:
            logger.error(f"Error analyzing historical invoices: {e}")
    
    def _analyze_historical_shipments(self):
        """Analyze historical shipment data to establish baselines"""
        try:
            # Load comprehensive shipment data
            shipment_file = f"{self.data_dir}/shipments/comprehensive_shipments.csv"
            if os.path.exists(shipment_file):
                df = pd.read_csv(shipment_file)
                
                # Analyze by route
                for _, row in df.iterrows():
                    route_key = f"{row['origin']} -> {row['destination']}"
                    
                    if route_key not in self.historical_data["route_profiles"]:
                        self.historical_data["route_profiles"][route_key] = {
                            "carriers": [],
                            "avg_transit_days": [],
                            "risk_scores": [],
                            "anomaly_types": []
                        }
                    
                    profile = self.historical_data["route_profiles"][route_key]
                    profile["carriers"].append(row['carrier'])
                    profile["risk_scores"].append(row.get('risk_score', 0.0))
                    profile["anomaly_types"].append(row.get('anomaly_type', 'none'))
                    
                    # Calculate transit days if dates available
                    if pd.notna(row.get('departure_date')) and pd.notna(row.get('estimated_arrival')):
                        try:
                            dep_date = pd.to_datetime(row['departure_date'])
                            arr_date = pd.to_datetime(row['estimated_arrival'])
                            transit_days = (arr_date - dep_date).days
                            profile["avg_transit_days"].append(transit_days)
                        except:
                            pass
                
                # Calculate route statistics
                for route_key, profile in self.historical_data["route_profiles"].items():
                    profile["common_carriers"] = list(set(profile["carriers"]))
                    profile["avg_risk_score"] = np.mean(profile["risk_scores"]) if profile["risk_scores"] else 0.0
                    profile["avg_transit_time"] = np.mean(profile["avg_transit_days"]) if profile["avg_transit_days"] else 7.0
                    profile["anomaly_rate"] = sum(1 for a in profile["anomaly_types"] if a != 'none') / len(profile["anomaly_types"])
                
                logger.info(f"Analyzed {len(df)} historical shipments for {len(self.historical_data['route_profiles'])} routes")
                
        except Exception as e:
            logger.error(f"Error analyzing historical shipments: {e}")
    
    def initialize_scoring_models(self):
        """Initialize ML-like scoring models for anomaly detection"""
        # Weights for different anomaly factors
        self.scoring_weights = {
            "invoice": {
                "amount_deviation": 0.25,
                "payment_terms_violation": 0.20,
                "timing_anomaly": 0.15,
                "supplier_risk": 0.15,
                "approval_missing": 0.10,
                "duplicate_risk": 0.10,
                "format_anomaly": 0.05
            },
            "shipment": {
                "route_deviation": 0.25,
                "carrier_risk": 0.20,
                "timing_anomaly": 0.20,
                "value_discrepancy": 0.15,
                "document_integrity": 0.10,
                "customs_delay": 0.10
            }
        }
    
    def detect_invoice_anomalies(self, invoice_data: Dict[str, Any]) -> List[AnomalyResult]:
        """Enhanced invoice anomaly detection with ML-like scoring"""
        anomalies = []
        
        # Extract key fields
        invoice_id = invoice_data.get("invoice_id", "unknown")
        supplier = invoice_data.get("supplier", "unknown")
        amount = float(invoice_data.get("amount", 0.0))
        issue_date = invoice_data.get("issue_date", "")
        due_date = invoice_data.get("due_date", "")
        payment_terms = invoice_data.get("payment_terms", "NET30")
        status = invoice_data.get("status", "pending")
        
        # Get supplier profile
        supplier_profile = self.historical_data["supplier_profiles"].get(supplier, {})
        
        # 1. Amount deviation analysis
        amount_anomaly = self._analyze_invoice_amount(invoice_id, supplier, amount, supplier_profile)
        if amount_anomaly:
            anomalies.append(amount_anomaly)
        
        # 2. Payment terms compliance
        terms_anomaly = self._analyze_payment_terms(invoice_id, supplier, payment_terms, issue_date, due_date)
        if terms_anomaly:
            anomalies.append(terms_anomaly)
        
        # 3. Timing analysis
        timing_anomaly = self._analyze_invoice_timing(invoice_id, issue_date, due_date)
        if timing_anomaly:
            anomalies.append(timing_anomaly)
        
        # 4. Duplicate detection
        duplicate_anomaly = self._analyze_invoice_duplicates(invoice_id, supplier, amount, issue_date)
        if duplicate_anomaly:
            anomalies.append(duplicate_anomaly)
        
        # 5. Approval workflow analysis
        approval_anomaly = self._analyze_invoice_approval(invoice_id, amount, invoice_data.get("approver"))
        if approval_anomaly:
            anomalies.append(approval_anomaly)
        
        # 6. Fraud pattern detection
        fraud_anomaly = self._analyze_invoice_fraud_patterns(invoice_id, invoice_data)
        if fraud_anomaly:
            anomalies.append(fraud_anomaly)
        
        return anomalies
    
    def _analyze_invoice_amount(self, invoice_id: str, supplier: str, amount: float, supplier_profile: Dict) -> Optional[AnomalyResult]:
        """Analyze invoice amount for anomalies"""
        if not supplier_profile or amount <= 0:
            return None
        
        avg_amount = supplier_profile.get("avg_amount", amount)
        std_amount = supplier_profile.get("std_amount", 0)
        
        if avg_amount > 0 and std_amount > 0:
            # Calculate z-score
            z_score = abs(amount - avg_amount) / std_amount
            
            # High z-score indicates potential anomaly
            if z_score > 2.5:  # More than 2.5 standard deviations
                risk_score = min(0.95, z_score / 5.0)  # Normalize to 0-0.95
                
                evidence = [
                    f"Invoice amount: ${amount:,.2f}",
                    f"Supplier average: ${avg_amount:,.2f}",
                    f"Standard deviation: ${std_amount:,.2f}",
                    f"Z-score: {z_score:.2f}"
                ]
                
                recommendations = [
                    "Verify invoice details with supplier",
                    "Check for additional services or bulk orders",
                    "Require additional approval for processing"
                ]
                
                severity = "high" if z_score > 4.0 else "medium" if z_score > 3.0 else "low"
                
                return AnomalyResult(
                    id=f"invoice_amount_{invoice_id}_{int(datetime.now().timestamp())}",
                    document_id=invoice_id,
                    anomaly_type="invoice_amount_deviation",
                    risk_score=risk_score,
                    severity=severity,
                    description=f"Invoice amount (${amount:,.2f}) deviates significantly from supplier's historical average (${avg_amount:,.2f})",
                    evidence=evidence,
                    recommendations=recommendations,
                    timestamp=datetime.now().timestamp(),
                    metadata={
                        "supplier": supplier,
                        "amount": amount,
                        "avg_amount": avg_amount,
                        "z_score": z_score,
                        "deviation_percentage": abs(amount - avg_amount) / avg_amount * 100
                    }
                )
        
        return None
    
    def _analyze_payment_terms(self, invoice_id: str, supplier: str, payment_terms: str, issue_date: str, due_date: str) -> Optional[AnomalyResult]:
        """Analyze payment terms compliance"""
        if not issue_date or not due_date:
            return None
        
        try:
            issue_dt = pd.to_datetime(issue_date)
            due_dt = pd.to_datetime(due_date)
            actual_days = (due_dt - issue_dt).days
            
            # Get expected terms for supplier
            supplier_terms = self.config["approved_entities"]["suppliers"].get(supplier, {})
            expected_terms = supplier_terms.get("payment_terms", "NET30")
            
            # Extract days from terms (e.g., "NET30" -> 30)
            expected_days = int(re.findall(r'\d+', expected_terms)[0]) if re.findall(r'\d+', expected_terms) else 30
            
            # Check for significant deviation
            deviation = abs(actual_days - expected_days)
            
            if deviation > self.config["invoice_rules"]["payment_terms_tolerance"]:
                risk_score = min(0.9, deviation / expected_days)
                
                evidence = [
                    f"Actual payment terms: {actual_days} days",
                    f"Expected terms for {supplier}: {expected_days} days",
                    f"Deviation: {deviation} days",
                    f"Issue date: {issue_date}",
                    f"Due date: {due_date}"
                ]
                
                recommendations = [
                    "Verify payment terms with supplier agreement",
                    "Check for special payment arrangements",
                    "Update supplier payment terms if needed"
                ]
                
                return AnomalyResult(
                    id=f"payment_terms_{invoice_id}_{int(datetime.now().timestamp())}",
                    document_id=invoice_id,
                    anomaly_type="payment_terms_violation",
                    risk_score=risk_score,
                    severity="medium" if deviation > 5 else "low",
                    description=f"Payment terms ({actual_days} days) deviate from expected terms ({expected_days} days) for supplier {supplier}",
                    evidence=evidence,
                    recommendations=recommendations,
                    timestamp=datetime.now().timestamp(),
                    metadata={
                        "supplier": supplier,
                        "actual_days": actual_days,
                        "expected_days": expected_days,
                        "deviation": deviation
                    }
                )
                
        except Exception as e:
            logger.error(f"Error analyzing payment terms for {invoice_id}: {e}")
        
        return None
    
    def _analyze_invoice_timing(self, invoice_id: str, issue_date: str, due_date: str) -> Optional[AnomalyResult]:
        """Analyze invoice timing for suspicious patterns"""
        if not issue_date:
            return None
        
        try:
            issue_dt = pd.to_datetime(issue_date)
            
            # Check for weekend processing
            if issue_dt.weekday() >= 5:  # Saturday or Sunday
                return AnomalyResult(
                    id=f"timing_weekend_{invoice_id}_{int(datetime.now().timestamp())}",
                    document_id=invoice_id,
                    anomaly_type="weekend_processing",
                    risk_score=self.config["invoice_rules"]["weekend_processing_risk"],
                    severity="medium",
                    description=f"Invoice issued on weekend ({issue_dt.strftime('%A, %Y-%m-%d')})",
                    evidence=[
                        f"Issue date: {issue_date}",
                        f"Day of week: {issue_dt.strftime('%A')}",
                        "Weekend processing is unusual for business invoices"
                    ],
                    recommendations=[
                        "Verify invoice authenticity",
                        "Check for automated processing systems",
                        "Confirm with supplier if needed"
                    ],
                    timestamp=datetime.now().timestamp(),
                    metadata={
                        "issue_date": issue_date,
                        "day_of_week": issue_dt.strftime('%A'),
                        "weekend_flag": True
                    }
                )
                
            # Check for after-hours processing (if time is included)
            # This would require time information in the date field
            
        except Exception as e:
            logger.error(f"Error analyzing timing for {invoice_id}: {e}")
        
        return None
    
    def _analyze_invoice_duplicates(self, invoice_id: str, supplier: str, amount: float, issue_date: str) -> Optional[AnomalyResult]:
        """Detect potential duplicate invoices"""
        # Check against existing anomalies for similar patterns
        duplicate_risk = 0.0
        evidence = []
        
        for anomaly in self.anomalies:
            if (anomaly.get("metadata", {}).get("supplier") == supplier and 
                anomaly.get("metadata", {}).get("amount") == amount and
                anomaly.get("metadata", {}).get("issue_date") == issue_date):
                
                duplicate_risk = 0.85
                evidence.append(f"Similar invoice found: {anomaly.get('document_id')}")
                break
        
        if duplicate_risk > 0.5:
            return AnomalyResult(
                id=f"duplicate_{invoice_id}_{int(datetime.now().timestamp())}",
                document_id=invoice_id,
                anomaly_type="potential_duplicate",
                risk_score=duplicate_risk,
                severity="high",
                description=f"Potential duplicate invoice detected for supplier {supplier}",
                evidence=evidence + [
                    f"Amount: ${amount:,.2f}",
                    f"Issue date: {issue_date}",
                    f"Supplier: {supplier}"
                ],
                recommendations=[
                    "Check for duplicate invoice numbers",
                    "Verify with accounts payable",
                    "Contact supplier to confirm"
                ],
                timestamp=datetime.now().timestamp(),
                metadata={
                    "supplier": supplier,
                    "amount": amount,
                    "issue_date": issue_date,
                    "duplicate_risk": duplicate_risk
                }
            )
        
        return None
    
    def _analyze_invoice_approval(self, invoice_id: str, amount: float, approver: Optional[str]) -> Optional[AnomalyResult]:
        """Analyze invoice approval workflow"""
        if not approver or amount <= 0:
            return None
        
        # Determine required approval level
        required_level = "manager"
        if amount >= self.config["invoice_rules"]["high_risk_amount_threshold"]:
            required_level = "director"
        if amount >= 50000:  # CFO approval threshold
            required_level = "cfo"
        
        # Check if approver matches required level (simplified)
        approver_level = "manager"  # Default assumption
        if "director" in approver.lower() or "cfo" in approver.lower():
            approver_level = "director" if "director" in approver.lower() else "cfo"
        
        # Simple approval hierarchy check
        approval_levels = {"manager": 1, "director": 2, "cfo": 3}
        
        if approval_levels.get(approver_level, 1) < approval_levels.get(required_level, 1):
            return AnomalyResult(
                id=f"approval_{invoice_id}_{int(datetime.now().timestamp())}",
                document_id=invoice_id,
                anomaly_type="insufficient_approval",
                risk_score=0.8,
                severity="high",
                description=f"Invoice amount (${amount:,.2f}) requires {required_level} approval but approved by {approver_level}",
                evidence=[
                    f"Invoice amount: ${amount:,.2f}",
                    f"Approver: {approver}",
                    f"Required level: {required_level}",
                    f"Actual level: {approver_level}"
                ],
                recommendations=[
                    "Escalate to appropriate approval level",
                    "Review approval workflow",
                    "Update approval limits if needed"
                ],
                timestamp=datetime.now().timestamp(),
                metadata={
                    "amount": amount,
                    "approver": approver,
                    "required_level": required_level,
                    "approver_level": approver_level
                }
            )
        
        return None
    
    def _analyze_invoice_fraud_patterns(self, invoice_id: str, invoice_data: Dict[str, Any]) -> Optional[AnomalyResult]:
        """Detect potential fraud patterns in invoice"""
        fraud_indicators = []
        fraud_score = 0.0
        
        amount = float(invoice_data.get("amount", 0.0))
        
        # Check for round amounts (potential fraud indicator)
        if amount in self.config["fraud_patterns"]["invoice"]["round_amounts"]:
            fraud_indicators.append("Round amount pattern")
            fraud_score += 0.3
        
        # Check for missing critical fields
        required_fields = ["supplier", "issue_date", "due_date", "payment_terms"]
        missing_fields = [field for field in required_fields if not invoice_data.get(field)]
        
        if missing_fields:
            fraud_indicators.append(f"Missing fields: {', '.join(missing_fields)}")
            fraud_score += 0.2 * len(missing_fields)
        
        # Check supplier legitimacy
        supplier = invoice_data.get("supplier", "")
        supplier_info = self.config["approved_entities"]["suppliers"].get(supplier, {})
        if supplier_info.get("risk_level") == "high":
            fraud_indicators.append("High risk supplier")
            fraud_score += 0.5
        
        if fraud_score >= 0.4:
            return AnomalyResult(
                id=f"fraud_{invoice_id}_{int(datetime.now().timestamp())}",
                document_id=invoice_id,
                anomaly_type="fraud_pattern",
                risk_score=min(0.95, fraud_score),
                severity="high" if fraud_score >= 0.7 else "medium",
                description=f"Multiple fraud indicators detected in invoice",
                evidence=fraud_indicators,
                recommendations=[
                    "Conduct detailed fraud investigation",
                    "Verify supplier legitimacy",
                    "Check supporting documentation",
                    "Consider holding payment pending review"
                ],
                timestamp=datetime.now().timestamp(),
                metadata={
                    "fraud_score": fraud_score,
                    "indicators": fraud_indicators,
                    "supplier": supplier,
                    "amount": amount
                }
            )
        
        return None
    
    def detect_shipment_anomalies(self, shipment_data: Dict[str, Any]) -> List[AnomalyResult]:
        """Enhanced shipment anomaly detection"""
        anomalies = []
        
        # Extract key fields
        shipment_id = shipment_data.get("shipment_id", "unknown")
        origin = shipment_data.get("origin", "")
        destination = shipment_data.get("destination", "")
        carrier = shipment_data.get("carrier", "")
        departure_date = shipment_data.get("departure_date", "")
        estimated_arrival = shipment_data.get("estimated_arrival", "")
        actual_arrival = shipment_data.get("actual_arrival", "")
        status = shipment_data.get("status", "unknown")
        
        # Get route profile
        route_key = f"{origin} -> {destination}"
        route_profile = self.historical_data["route_profiles"].get(route_key, {})
        
        # 1. Carrier validation
        carrier_anomaly = self._analyze_shipment_carrier(shipment_id, carrier, route_profile)
        if carrier_anomaly:
            anomalies.append(carrier_anomaly)
        
        # 2. Route timing analysis
        timing_anomaly = self._analyze_shipment_timing(shipment_id, departure_date, estimated_arrival, actual_arrival, route_profile)
        if timing_anomaly:
            anomalies.append(timing_anomaly)
        
        # 3. Route deviation analysis
        route_anomaly = self._analyze_shipment_route(shipment_id, origin, destination, carrier, route_profile)
        if route_anomaly:
            anomalies.append(route_anomaly)
        
        # 4. Status consistency check
        status_anomaly = self._analyze_shipment_status(shipment_id, status, departure_date, estimated_arrival, actual_arrival)
        if status_anomaly:
            anomalies.append(status_anomaly)
        
        return anomalies
    
    def _analyze_shipment_carrier(self, shipment_id: str, carrier: str, route_profile: Dict) -> Optional[AnomalyResult]:
        """Analyze carrier selection for anomalies"""
        if not carrier:
            return None
        
        # Check against approved carriers
        all_approved = (self.config["approved_entities"]["carriers"]["tier_1"] + 
                       self.config["approved_entities"]["carriers"]["tier_2"])
        
        if carrier in self.config["approved_entities"]["carriers"]["blacklisted"]:
            return AnomalyResult(
                id=f"carrier_blacklisted_{shipment_id}_{int(datetime.now().timestamp())}",
                document_id=shipment_id,
                anomaly_type="blacklisted_carrier",
                risk_score=0.95,
                severity="high",
                description=f"Shipment using blacklisted carrier: {carrier}",
                evidence=[
                    f"Carrier: {carrier}",
                    "Carrier is on blacklist",
                    "High fraud risk"
                ],
                recommendations=[
                    "Immediately investigate shipment",
                    "Contact security team",
                    "Verify shipment authenticity",
                    "Consider blocking shipment"
                ],
                timestamp=datetime.now().timestamp(),
                metadata={
                    "carrier": carrier,
                    "blacklisted": True
                }
            )
        
        # Check against route history
        if route_profile and route_profile.get("common_carriers"):
            if carrier not in route_profile["common_carriers"]:
                risk_score = 0.6 if carrier in all_approved else 0.8
                
                return AnomalyResult(
                    id=f"carrier_unusual_{shipment_id}_{int(datetime.now().timestamp())}",
                    document_id=shipment_id,
                    anomaly_type="unusual_carrier",
                    risk_score=risk_score,
                    severity="medium",
                    description=f"Unusual carrier ({carrier}) for this route",
                    evidence=[
                        f"Carrier: {carrier}",
                        f"Common carriers for route: {', '.join(route_profile['common_carriers'])}",
                        f"Carrier approved: {'Yes' if carrier in all_approved else 'No'}"
                    ],
                    recommendations=[
                        "Verify carrier selection reason",
                        "Check carrier credentials",
                        "Monitor shipment closely"
                    ],
                    timestamp=datetime.now().timestamp(),
                    metadata={
                        "carrier": carrier,
                        "common_carriers": route_profile["common_carriers"],
                        "approved": carrier in all_approved
                    }
                )
        
        return None
    
    def _analyze_shipment_timing(self, shipment_id: str, departure_date: str, estimated_arrival: str, actual_arrival: str, route_profile: Dict) -> Optional[AnomalyResult]:
        """Analyze shipment timing for anomalies"""
        if not departure_date or not estimated_arrival:
            return None
        
        try:
            dep_dt = pd.to_datetime(departure_date)
            est_arr_dt = pd.to_datetime(estimated_arrival)
            
            estimated_days = (est_arr_dt - dep_dt).days
            
            # Compare with historical route performance
            if route_profile and route_profile.get("avg_transit_time"):
                avg_transit = route_profile["avg_transit_time"]
                deviation = abs(estimated_days - avg_transit)
                
                if deviation > self.config["shipment_rules"]["delivery_delay_days"]:
                    risk_score = min(0.9, deviation / avg_transit)
                    
                    evidence = [
                        f"Estimated transit time: {estimated_days} days",
                        f"Average for route: {avg_transit:.1f} days",
                        f"Deviation: {deviation:.1f} days"
                    ]
                    
                    # Check if actual arrival is available
                    if actual_arrival and actual_arrival != "null":
                        try:
                            act_arr_dt = pd.to_datetime(actual_arrival)
                            actual_days = (act_arr_dt - dep_dt).days
                            evidence.append(f"Actual transit time: {actual_days} days")
                        except:
                            pass
                    
                    return AnomalyResult(
                        id=f"timing_{shipment_id}_{int(datetime.now().timestamp())}",
                        document_id=shipment_id,
                        anomaly_type="transit_time_anomaly",
                        risk_score=risk_score,
                        severity="high" if deviation > 5 else "medium",
                        description=f"Transit time ({estimated_days} days) significantly deviates from route average ({avg_transit:.1f} days)",
                        evidence=evidence,
                        recommendations=[
                            "Investigate cause of delay",
                            "Check for route changes",
                            "Verify carrier performance",
                            "Consider alternative routing"
                        ],
                        timestamp=datetime.now().timestamp(),
                        metadata={
                            "estimated_days": estimated_days,
                            "avg_transit": avg_transit,
                            "deviation": deviation,
                            "departure_date": departure_date,
                            "estimated_arrival": estimated_arrival
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Error analyzing timing for {shipment_id}: {e}")
        
        return None
    
    def _analyze_shipment_route(self, shipment_id: str, origin: str, destination: str, carrier: str, route_profile: Dict) -> Optional[AnomalyResult]:
        """Analyze route for potential deviations or inefficiencies"""
        if not origin or not destination:
            return None
        
        # Basic route validation
        route_key = f"{origin} -> {destination}"
        
        # Check for suspicious route patterns
        suspicious_patterns = [
            "unknown -> unknown",
            "suspicious -> suspicious",
            "fake -> fake"
        ]
        
        if any(pattern in route_key.lower() for pattern in suspicious_patterns):
            return AnomalyResult(
                id=f"route_suspicious_{shipment_id}_{int(datetime.now().timestamp())}",
                document_id=shipment_id,
                anomaly_type="suspicious_route",
                risk_score=0.85,
                severity="high",
                description=f"Suspicious route detected: {route_key}",
                evidence=[
                    f"Origin: {origin}",
                    f"Destination: {destination}",
                    f"Carrier: {carrier}",
                    "Route matches suspicious pattern"
                ],
                recommendations=[
                    "Verify route authenticity",
                    "Check origin and destination validity",
                    "Investigate carrier credentials",
                    "Consider route blocking"
                ],
                timestamp=datetime.now().timestamp(),
                metadata={
                    "origin": origin,
                    "destination": destination,
                    "carrier": carrier,
                    "route_key": route_key
                }
            )
        
        return None
    
    def _analyze_shipment_status(self, shipment_id: str, status: str, departure_date: str, estimated_arrival: str, actual_arrival: str) -> Optional[AnomalyResult]:
        """Analyze shipment status for consistency"""
        if not status:
            return None
        
        try:
            current_date = datetime.now()
            
            # Check status consistency with dates
            if departure_date:
                dep_dt = pd.to_datetime(departure_date)
                
                # If departure date is in the future but status is "In Transit"
                if dep_dt > current_date and status.lower() == "in transit":
                    return AnomalyResult(
                        id=f"status_inconsistent_{shipment_id}_{int(datetime.now().timestamp())}",
                        document_id=shipment_id,
                        anomaly_type="status_inconsistency",
                        risk_score=0.7,
                        severity="medium",
                        description=f"Status '{status}' inconsistent with future departure date",
                        evidence=[
                            f"Status: {status}",
                            f"Departure date: {departure_date}",
                            f"Current date: {current_date.strftime('%Y-%m-%d')}",
                            "Departure date is in the future"
                        ],
                        recommendations=[
                            "Verify departure date accuracy",
                            "Update status if needed",
                            "Check for data entry errors"
                        ],
                        timestamp=datetime.now().timestamp(),
                        metadata={
                            "status": status,
                            "departure_date": departure_date,
                            "future_departure": True
                        }
                    )
            
            # Check if shipment is overdue
            if estimated_arrival and status.lower() in ["in transit", "pending"]:
                est_arr_dt = pd.to_datetime(estimated_arrival)
                
                if est_arr_dt < current_date:
                    days_overdue = (current_date - est_arr_dt).days
                    
                    if days_overdue > 1:  # Allow 1 day grace period
                        return AnomalyResult(
                            id=f"overdue_{shipment_id}_{int(datetime.now().timestamp())}",
                            document_id=shipment_id,
                            anomaly_type="shipment_overdue",
                            risk_score=min(0.9, days_overdue / 10.0),
                            severity="high" if days_overdue > 5 else "medium",
                            description=f"Shipment is {days_overdue} days overdue",
                            evidence=[
                                f"Status: {status}",
                                f"Estimated arrival: {estimated_arrival}",
                                f"Current date: {current_date.strftime('%Y-%m-%d')}",
                                f"Days overdue: {days_overdue}"
                            ],
                            recommendations=[
                                "Contact carrier for status update",
                                "Investigate potential delays",
                                "Update customer on delivery status",
                                "Consider alternative shipping"
                            ],
                            timestamp=datetime.now().timestamp(),
                            metadata={
                                "status": status,
                                "estimated_arrival": estimated_arrival,
                                "days_overdue": days_overdue
                            }
                        )
                        
        except Exception as e:
            logger.error(f"Error analyzing status for {shipment_id}: {e}")
        
        return None
    
    def load_existing_anomalies(self) -> List[Dict[str, Any]]:
        """Load existing anomalies from storage"""
        anomalies_file = f"{self.anomalies_dir}/anomalies.json"
        
        if os.path.exists(anomalies_file):
            try:
                with open(anomalies_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading anomalies: {e}")
        
        return []
    
    def save_anomalies(self, anomalies: List[AnomalyResult]):
        """Save anomalies to storage"""
        anomalies_file = f"{self.anomalies_dir}/anomalies.json"
        
        # Convert AnomalyResult objects to dictionaries
        anomaly_dicts = []
        for anomaly in anomalies:
            anomaly_dict = {
                "id": anomaly.id,
                "document_id": anomaly.document_id,
                "anomaly_type": anomaly.anomaly_type,
                "risk_score": anomaly.risk_score,
                "severity": anomaly.severity,
                "description": anomaly.description,
                "evidence": anomaly.evidence,
                "recommendations": anomaly.recommendations,
                "timestamp": anomaly.timestamp,
                "metadata": anomaly.metadata
            }
            anomaly_dicts.append(anomaly_dict)
        
        # Add to existing anomalies
        self.anomalies.extend(anomaly_dicts)
        
        try:
            with open(anomalies_file, 'w') as f:
                json.dump(self.anomalies, f, indent=2)
            logger.info(f"Saved {len(anomalies)} new anomalies")
        except Exception as e:
            logger.error(f"Error saving anomalies: {e}")
    
    def process_document(self, doc_path: str, doc_type: str, doc_data: Dict[str, Any]) -> List[AnomalyResult]:
        """Process a document and detect anomalies"""
        anomalies = []
        
        try:
            if doc_type == "invoice":
                anomalies = self.detect_invoice_anomalies(doc_data)
            elif doc_type == "shipment":
                anomalies = self.detect_shipment_anomalies(doc_data)
            
            if anomalies:
                self.save_anomalies(anomalies)
                logger.info(f"Detected {len(anomalies)} anomalies in {doc_type} document {doc_path}")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
            return []
    
    def get_anomalies_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies"""
        if not self.anomalies:
            return {
                "total_anomalies": 0,
                "by_type": {},
                "by_severity": {},
                "high_risk_count": 0
            }
        
        summary = {
            "total_anomalies": len(self.anomalies),
            "by_type": {},
            "by_severity": {},
            "high_risk_count": 0
        }
        
        for anomaly in self.anomalies:
            # Count by type
            anomaly_type = anomaly.get("anomaly_type", "unknown")
            summary["by_type"][anomaly_type] = summary["by_type"].get(anomaly_type, 0) + 1
            
            # Count by severity
            severity = anomaly.get("severity", "unknown")
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
            
            # Count high-risk anomalies
            if anomaly.get("risk_score", 0) >= 0.7:
                summary["high_risk_count"] += 1
        
        return summary
