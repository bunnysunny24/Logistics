import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger
import re

class AnomalyDetector:
    def __init__(self):
        # Enhanced thresholds for anomaly detection
        self.route_deviation_threshold = 100  # miles
        self.timeline_variation_threshold = {
            "air": 48,  # hours - increased for better accuracy
            "ocean": 120,  # hours - increased for better accuracy
            "ground": 24  # hours - added ground transportation
        }
        self.value_discrepancy_threshold = 0.15  # 15% - more realistic threshold
        
        # Invoice-specific thresholds
        self.invoice_amount_threshold = 0.20  # 20% variance threshold
        self.payment_terms_tolerance = 2  # days tolerance
        
        # Approval thresholds based on policies
        self.approval_thresholds = {
            "manager": 5000,     # $5K requires manager approval
            "director": 25000,   # $25K requires director approval
            "cfo": 50000        # $50K requires CFO approval
        }
        
        # Carrier validation based on policies
        self.approved_carriers = {
            "air": ["Global Shipping Inc", "Air Express International", "WorldWide Cargo"],
            "ocean": ["Ocean Express", "Global Shipping Inc", "Asia Logistics", "Maritime Transporters"],
            "ground": ["Nationwide Express", "Regional Carriers Inc"]
        }
        
        # Supplier-specific payment terms from policies
        self.supplier_payment_terms = {
            "ABC Electronics": {"terms": "NET30", "early_discount": 0.02, "late_penalty": 0.015},
            "Global Logistics": {"terms": "NET15", "early_discount": 0.00, "late_penalty": 0.02},
            "Tech Solutions Inc": {"terms": "NET45", "early_discount": 0.01, "late_penalty": 0.015},
            "FastTrack Shipping": {"terms": "NET15", "early_discount": 0.00, "late_penalty": 0.03},
            "Quality Parts Co": {"terms": "NET30", "early_discount": 0.015, "late_penalty": 0.015}
        }
        
        # Historical data storage
        self.historical_data = {
            "shipments": {},
            "invoices": {}
        }
        
        # Anomaly registry with enhanced structure
        self.anomalies = []
        self.processed_documents = set()  # Track processed documents to avoid duplicates
        
        # Load historical data and policies
        self._load_historical_data()
        self._load_policy_data()
    
    def _load_historical_data(self):
        """Load historical data from files and comprehensive datasets"""
        try:
            # Load comprehensive shipment data
            comprehensive_shipments_path = os.path.join("data", "shipments", "comprehensive_shipments.csv")
            if os.path.exists(comprehensive_shipments_path):
                df = pd.read_csv(comprehensive_shipments_path)
                self._process_shipment_historical_data(df)
                logger.info(f"Loaded {len(df)} comprehensive shipment records")
            
            # Load comprehensive invoice data
            comprehensive_invoices_path = os.path.join("data", "invoices", "comprehensive_invoices.csv")
            if os.path.exists(comprehensive_invoices_path):
                df = pd.read_csv(comprehensive_invoices_path)
                self._process_invoice_historical_data(df)
                logger.info(f"Loaded {len(df)} comprehensive invoice records")
            
            # Load individual shipment files for detailed analysis
            shipment_files = ["shipment_001.csv", "shipment_002.csv", "shipment_003_abnormal.csv", "shipment_004_abnormal.csv"]
            for file in shipment_files:
                file_path = os.path.join("data", "shipments", file)
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        if len(df) > 0:
                            self._process_individual_shipment_data(df, file)
                    except Exception as e:
                        logger.warning(f"Could not process {file}: {e}")
            
            # Load individual invoice files for detailed analysis
            invoice_files = ["invoice_001.csv", "invoice_002.csv", "invoice_003.csv", "invoice_004_abnormal.csv"]
            for file in invoice_files:
                file_path = os.path.join("data", "invoices", file)
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        if len(df) > 0:
                            self._process_individual_invoice_data(df, file)
                    except Exception as e:
                        logger.warning(f"Could not process {file}: {e}")
                        
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def _process_shipment_historical_data(self, df):
        """Process comprehensive shipment data for historical patterns"""
        for _, row in df.iterrows():
            origin = row.get('origin', '')
            destination = row.get('destination', '')
            carrier = row.get('carrier', '')
            
            if origin and destination and carrier:
                key = f"{origin}_{destination}_{carrier}"
                
                # Calculate transit time if available
                transit_time = None
                if pd.notna(row.get('departure_date')) and pd.notna(row.get('estimated_arrival')):
                    try:
                        dep_date = pd.to_datetime(row['departure_date'])
                        arr_date = pd.to_datetime(row['estimated_arrival'])
                        transit_time = (arr_date - dep_date).total_seconds() / 3600  # hours
                    except:
                        pass
                
                if key not in self.historical_data["shipments"]:
                    self.historical_data["shipments"][key] = {
                        "transit_times": [],
                        "risk_scores": [],
                        "route_info": {}
                    }
                
                if transit_time:
                    self.historical_data["shipments"][key]["transit_times"].append(transit_time)
                
                if pd.notna(row.get('risk_score')):
                    self.historical_data["shipments"][key]["risk_scores"].append(float(row['risk_score']))
    
    def _process_invoice_historical_data(self, df):
        """Process comprehensive invoice data for historical patterns"""
        for _, row in df.iterrows():
            supplier = row.get('supplier', '')
            payment_terms = row.get('payment_terms', '')
            
            if supplier and payment_terms:
                key = f"{supplier}_{payment_terms}"
                
                if key not in self.historical_data["invoices"]:
                    self.historical_data["invoices"][key] = {
                        "amounts": [],
                        "early_discounts": [],
                        "approvers": set()
                    }
                
                if pd.notna(row.get('amount')):
                    self.historical_data["invoices"][key]["amounts"].append(float(row['amount']))
                
                if pd.notna(row.get('early_discount')):
                    self.historical_data["invoices"][key]["early_discounts"].append(float(row['early_discount']))
                
                if pd.notna(row.get('approver')):
                    self.historical_data["invoices"][key]["approvers"].add(row['approver'])
    
    def _process_individual_shipment_data(self, df, filename):
        """Process individual shipment files for specific patterns"""
        # Extract specific anomaly patterns from individual files
        if "abnormal" in filename and len(df) > 1:
            # Look for detailed item information
            item_rows = df[df.columns[0] == 'item']
            if not item_rows.empty:
                # Process item-level data for value discrepancy detection
                logger.debug(f"Found detailed shipment data in {filename}")
    
    def _process_individual_invoice_data(self, df, filename):
        """Process individual invoice files for specific patterns"""
        # Extract specific anomaly patterns from individual files
        if "abnormal" in filename and len(df) > 1:
            # Look for detailed item information
            item_rows = df[df.columns[0] == 'item']
            if not item_rows.empty:
                # Process item-level data for amount validation
                logger.debug(f"Found detailed invoice data in {filename}")
    
    def _load_policy_data(self):
        """Load policy data from markdown files"""
        try:
            # Load shipment guidelines
            shipment_policy_path = os.path.join("data", "policies", "shipment-guidelines-v2.md")
            if os.path.exists(shipment_policy_path):
                with open(shipment_policy_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract carrier information using regex
                    self._extract_carrier_info(content)
                    logger.info("Loaded shipment policy data")
            
            # Load payment rules
            payment_policy_path = os.path.join("data", "policies", "payout-rules-v3.md")
            if os.path.exists(payment_policy_path):
                with open(payment_policy_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract payment terms using regex
                    self._extract_payment_terms(content)
                    logger.info("Loaded payment policy data")
                    
        except Exception as e:
            logger.error(f"Error loading policy data: {e}")
    
    def _extract_carrier_info(self, content):
        """Extract carrier information from policy markdown"""
        # This is a simplified extraction - in production, you'd want more robust parsing
        lines = content.split('\n')
        current_mode = None
        
        for line in lines:
            if "International Air Freight" in line:
                current_mode = "air"
            elif "International Ocean Freight" in line:
                current_mode = "ocean"
            elif "Domestic Ground" in line:
                current_mode = "ground"
            elif line.startswith('- ') and current_mode:
                carrier_name = line.strip('- ').split('\n')[0]
                if carrier_name and current_mode in self.approved_carriers:
                    if carrier_name not in self.approved_carriers[current_mode]:
                        self.approved_carriers[current_mode].append(carrier_name)
    
    def _extract_payment_terms(self, content):
        """Extract payment terms from policy markdown"""
        # Extract supplier-specific terms using regex patterns
        supplier_sections = re.findall(r'### \d+\.\d+ (.+?)\n(.*?)(?=### |\Z)', content, re.DOTALL)
        
        for supplier_name, section in supplier_sections:
            if supplier_name in self.supplier_payment_terms:
                # Extract payment terms
                terms_match = re.search(r'Payment terms: (NET\d+)', section)
                discount_match = re.search(r'Early payment discount: (\d+(?:\.\d+)?)%', section)
                penalty_match = re.search(r'Late payment penalty: (\d+(?:\.\d+)?)%', section)
                
                if terms_match:
                    self.supplier_payment_terms[supplier_name]["terms"] = terms_match.group(1)
                if discount_match:
                    self.supplier_payment_terms[supplier_name]["early_discount"] = float(discount_match.group(1)) / 100
                if penalty_match:
                    self.supplier_payment_terms[supplier_name]["late_penalty"] = float(penalty_match.group(1)) / 100
    
    def detect_shipment_anomalies(self, shipment_data):
        """Detect anomalies in shipment data"""
        anomalies = []
        
        # Get shipment ID
        shipment_id = shipment_data.get("shipment_id")
        if not shipment_id:
            logger.warning("Shipment ID missing in data")
            return anomalies
        
        # Check for route deviation
        route_anomaly = self._check_route_deviation(shipment_data)
        if route_anomaly:
            anomalies.append(route_anomaly)
        
        # Check for timeline variation
        timeline_anomaly = self._check_timeline_variation(shipment_data)
        if timeline_anomaly:
            anomalies.append(timeline_anomaly)
        
        # Check for value discrepancy
        value_anomaly = self._check_value_discrepancy(shipment_data)
        if value_anomaly:
            anomalies.append(value_anomaly)
        
        # Check for duplicate shipment IDs
        duplicate_anomaly = self._check_duplicate_shipment(shipment_data)
        if duplicate_anomaly:
            anomalies.append(duplicate_anomaly)
        
        # Register the anomalies
        for anomaly in anomalies:
            self.anomalies.append(anomaly)
        
        return anomalies
    
    def detect_invoice_anomalies(self, invoice_data):
        """Detect anomalies in invoice data"""
        anomalies = []
        
        # Get invoice ID
        invoice_id = invoice_data.get("invoice_id")
        if not invoice_id:
            logger.warning("Invoice ID missing in data")
            return anomalies
        
        # Check for payment terms compliance
        terms_anomaly = self._check_payment_terms_compliance(invoice_data)
        if terms_anomaly:
            anomalies.append(terms_anomaly)
        
        # Check for amount discrepancy
        amount_anomaly = self._check_invoice_amount_discrepancy(invoice_data)
        if amount_anomaly:
            anomalies.append(amount_anomaly)
        
        # Check for approval workflow compliance
        approval_anomaly = self._check_approval_workflow_compliance(invoice_data)
        if approval_anomaly:
            anomalies.append(approval_anomaly)
        
        # Register the anomalies
        for anomaly in anomalies:
            self.anomalies.append(anomaly)
        
        return anomalies
    
    def _check_route_deviation(self, shipment_data):
        """Check for route deviation in shipment"""
        origin = shipment_data.get("origin")
        destination = shipment_data.get("destination")
        carrier = shipment_data.get("carrier")
        actual_route = shipment_data.get("actual_route")
        
        if not (origin and destination and carrier):
            return None
        
        # Create lookup key
        key = f"{origin}_{destination}_{carrier}"
        
        # Check if we have historical data for this route
        if key in self.historical_data["shipments"]:
            historical = self.historical_data["shipments"][key][0]  # Get first record
            expected_route = historical.get("expected_route")
            
            if expected_route and actual_route and expected_route != actual_route:
                # Calculate a simple risk score based on deviation
                risk_score = 0.8  # Default high risk
                
                # Create anomaly record
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                    "document_id": shipment_data.get("shipment_id"),
                    "anomaly_type": "route_deviation",
                    "description": f"Shipment route deviates from expected path. Expected: {expected_route}, Actual: {actual_route}",
                    "risk_score": risk_score,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "origin": origin,
                        "destination": destination,
                        "carrier": carrier,
                        "expected_route": expected_route,
                        "actual_route": actual_route
                    }
                }
        
        return None
    
    def _check_timeline_variation(self, shipment_data):
        """Check for timeline variation in shipment"""
        estimated_arrival = shipment_data.get("estimated_arrival")
        actual_arrival = shipment_data.get("actual_arrival")
        status = shipment_data.get("status")
        
        # If shipment is delivered and we have both dates
        if status == "Delayed" or (status == "Delivered" and estimated_arrival and actual_arrival):
            try:
                # Convert to datetime objects
                est_date = datetime.strptime(estimated_arrival, "%Y-%m-%d")
                
                if actual_arrival and actual_arrival != "null":
                    act_date = datetime.strptime(actual_arrival, "%Y-%m-%d")
                    
                    # Calculate delay in hours
                    delay_hours = (act_date - est_date).total_seconds() / 3600
                    
                    # Determine freight type (simplistic approach)
                    freight_type = "air" if delay_hours < 48 else "ocean"
                    
                    # Check if delay exceeds threshold
                    if delay_hours > self.timeline_variation_threshold[freight_type]:
                        # Calculate risk score
                        if freight_type == "air":
                            risk_score = min(0.9, 0.4 + (delay_hours / 72) * 0.5)
                        else:  # ocean
                            risk_score = min(0.9, 0.4 + (delay_hours / 168) * 0.5)
                        
                        # Create anomaly record
                        return {
                            "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                            "document_id": shipment_data.get("shipment_id"),
                            "anomaly_type": "timeline_delay",
                            "description": f"Shipment arrived {int(delay_hours)} hours later than estimated.",
                            "risk_score": risk_score,
                            "timestamp": int(datetime.now().timestamp()),
                            "metadata": {
                                "estimated_arrival": estimated_arrival,
                                "actual_arrival": actual_arrival,
                                "delay_hours": int(delay_hours),
                                "freight_type": freight_type
                            }
                        }
                elif status == "Delayed":
                    # For delayed shipments without actual arrival
                    current_date = datetime.now()
                    delay_hours = (current_date - est_date).total_seconds() / 3600
                    
                    # Determine freight type (simplistic approach)
                    freight_type = "air" if delay_hours < 48 else "ocean"
                    
                    # Check if delay exceeds threshold
                    if delay_hours > self.timeline_variation_threshold[freight_type]:
                        # Calculate risk score
                        if freight_type == "air":
                            risk_score = min(0.9, 0.4 + (delay_hours / 72) * 0.5)
                        else:  # ocean
                            risk_score = min(0.9, 0.4 + (delay_hours / 168) * 0.5)
                        
                        # Create anomaly record
                        return {
                            "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                            "document_id": shipment_data.get("shipment_id"),
                            "anomaly_type": "timeline_delay",
                            "description": f"Shipment is currently {int(delay_hours)} hours delayed.",
                            "risk_score": risk_score,
                            "timestamp": int(datetime.now().timestamp()),
                            "metadata": {
                                "estimated_arrival": estimated_arrival,
                                "current_delay_hours": int(delay_hours),
                                "freight_type": freight_type
                            }
                        }
            except Exception as e:
                logger.error(f"Error checking timeline variation: {e}")
        
        return None
    
    def _check_value_discrepancy(self, shipment_data):
        """Check for value discrepancy in shipment"""
        origin = shipment_data.get("origin")
        destination = shipment_data.get("destination")
        carrier = shipment_data.get("carrier")
        declared_value = shipment_data.get("value")
        
        if not (origin and destination and carrier and declared_value):
            return None
        
        try:
            # Convert value to float
            declared_value = float(declared_value)
            
            # Create lookup key
            key = f"{origin}_{destination}_{carrier}"
            
            # Check if we have historical data for this route
            if key in self.historical_data["shipments"]:
                historical = self.historical_data["shipments"][key][0]  # Get first record
                avg_value = historical.get("avg_value")
                std_value = historical.get("std_value")
                
                if avg_value:
                    # Calculate z-score
                    z_score = abs(declared_value - avg_value) / (std_value if std_value else avg_value * 0.1)
                    
                    # Calculate percentage difference
                    pct_diff = abs(declared_value - avg_value) / avg_value
                    
                    # Check if difference exceeds threshold
                    if pct_diff > self.value_discrepancy_threshold:
                        # Calculate risk score based on z-score
                        risk_score = min(0.95, 0.5 + min(z_score / 10, 0.45))
                        
                        # Create anomaly record
                        return {
                            "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                            "document_id": shipment_data.get("shipment_id"),
                            "anomaly_type": "value_discrepancy",
                            "description": f"Shipment value differs by {pct_diff:.1%} from historical average.",
                            "risk_score": risk_score,
                            "timestamp": int(datetime.now().timestamp()),
                            "metadata": {
                                "declared_value": declared_value,
                                "historical_avg_value": avg_value,
                                "percentage_difference": pct_diff,
                                "z_score": z_score
                            }
                        }
        except Exception as e:
            logger.error(f"Error checking value discrepancy: {e}")
        
        return None
    
    def _check_duplicate_shipment(self, shipment_data):
        """Check for duplicate shipment IDs"""
        shipment_id = shipment_data.get("shipment_id")
        
        # Simplistic approach - check if we've seen this shipment ID before
        # In a real system, this would check against a database
        for anomaly in self.anomalies:
            if (anomaly["document_id"] == shipment_id and 
                anomaly["anomaly_type"] != "duplicate_id"):
                
                # Create anomaly record
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                    "document_id": shipment_id,
                    "anomaly_type": "duplicate_id",
                    "description": f"Duplicate shipment ID detected: {shipment_id}",
                    "risk_score": 0.9,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "original_timestamp": anomaly["timestamp"]
                    }
                }
        
        return None
    
    def _check_payment_terms_compliance(self, invoice_data):
        """Check for payment terms compliance in invoice"""
        supplier = invoice_data.get("supplier")
        payment_terms = invoice_data.get("payment_terms")
        issue_date = invoice_data.get("issue_date")
        due_date = invoice_data.get("due_date")
        
        if not (supplier and payment_terms and issue_date and due_date):
            return None
        
        try:
            # Convert dates to datetime objects
            issue_dt = datetime.strptime(issue_date, "%Y-%m-%d")
            due_dt = datetime.strptime(due_date, "%Y-%m-%d")
            
            # Calculate days between issue and due dates
            days_diff = (due_dt - issue_dt).days
            
            # Extract the number from payment terms (e.g., "NET30" -> 30)
            expected_days = int(payment_terms.replace("NET", ""))
            
            # Check if due date is compliant with payment terms
            if abs(days_diff - expected_days) > 1:  # Allow 1 day tolerance
                # Calculate risk score
                risk_score = 0.7  # Default high risk for non-compliance
                
                # Create anomaly record
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                    "document_id": invoice_data.get("invoice_id"),
                    "anomaly_type": "payment_terms_noncompliance",
                    "description": f"Invoice due date doesn't match payment terms. Expected {expected_days} days, got {days_diff} days.",
                    "risk_score": risk_score,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "supplier": supplier,
                        "payment_terms": payment_terms,
                        "issue_date": issue_date,
                        "due_date": due_date,
                        "expected_days": expected_days,
                        "actual_days": days_diff
                    }
                }
        except Exception as e:
            logger.error(f"Error checking payment terms compliance: {e}")
        
        return None
    
    def _check_invoice_amount_discrepancy(self, invoice_data):
        """Check for amount discrepancy in invoice"""
        supplier = invoice_data.get("supplier")
        payment_terms = invoice_data.get("payment_terms")
        amount = invoice_data.get("amount")
        
        if not (supplier and payment_terms and amount):
            return None
        
        try:
            # Convert amount to float
            amount = float(amount)
            
            # Create lookup key
            key = f"{supplier}_{payment_terms}"
            
            # Check if we have historical data for this supplier
            if key in self.historical_data["invoices"]:
                historical = self.historical_data["invoices"][key][0]  # Get first record
                avg_amount = historical.get("avg_amount")
                std_amount = historical.get("std_amount")
                
                if avg_amount:
                    # Calculate z-score
                    z_score = abs(amount - avg_amount) / (std_amount if std_amount else avg_amount * 0.1)
                    
                    # Calculate percentage difference
                    pct_diff = abs(amount - avg_amount) / avg_amount
                    
                    # Check if difference exceeds threshold
                    if pct_diff > 0.25:  # 25% threshold for invoices
                        # Calculate risk score based on z-score
                        risk_score = min(0.9, 0.5 + min(z_score / 10, 0.4))
                        
                        # Create anomaly record
                        return {
                            "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                            "document_id": invoice_data.get("invoice_id"),
                            "anomaly_type": "invoice_amount_discrepancy",
                            "description": f"Invoice amount differs by {pct_diff:.1%} from historical average for this supplier.",
                            "risk_score": risk_score,
                            "timestamp": int(datetime.now().timestamp()),
                            "metadata": {
                                "amount": amount,
                                "historical_avg_amount": avg_amount,
                                "percentage_difference": pct_diff,
                                "z_score": z_score
                            }
                        }
        except Exception as e:
            logger.error(f"Error checking invoice amount discrepancy: {e}")
        
        return None
    
    def _check_approval_workflow_compliance(self, invoice_data):
        """Check for approval workflow compliance in invoice"""
        amount = invoice_data.get("amount")
        approver = invoice_data.get("approver")
        
        if not (amount and approver):
            return None
        
        try:
            # Convert amount to float
            amount = float(amount)
            
            # Determine required approval level
            required_approvers = []
            if amount < 5000:
                required_approvers = ["department_manager"]
            elif 5000 <= amount <= 25000:
                required_approvers = ["department_manager", "finance_director"]
            else:  # amount > 25000
                required_approvers = ["department_manager", "finance_director", "cfo"]
            
            # Check if approver matches required level
            # This is a simplified check - in reality would verify against a database
            if len(required_approvers) > 1 and not (
                "," in approver or  # Multiple approvers separated by comma
                "director" in approver.lower() or  # Finance director
                "cfo" in approver.lower()  # CFO
            ):
                # Calculate risk score
                risk_score = 0.8 if amount > 25000 else 0.6
                
                # Create anomaly record
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                    "document_id": invoice_data.get("invoice_id"),
                    "anomaly_type": "approval_workflow_noncompliance",
                    "description": f"Invoice requires {len(required_approvers)} level(s) of approval but only has one approver.",
                    "risk_score": risk_score,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "amount": amount,
                        "current_approver": approver,
                        "required_approvers": required_approvers,
                        "approval_threshold": "$5,000-$25,000" if 5000 <= amount <= 25000 else ">$25,000"
                    }
                }
        except Exception as e:
            logger.error(f"Error checking approval workflow compliance: {e}")
        
        return None
    
    def get_anomalies(self, min_risk_score=0.5, start_date=None, end_date=None):
        """Get anomalies with filtering"""
        filtered_anomalies = []
        
        # Convert date strings to timestamps if provided
        start_ts = None
        end_ts = None
        
        if start_date:
            try:
                start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            def _load_policy_data(self):
        """Load policy data from markdown files"""
        try:
            # Load shipment guidelines
            shipment_policy_path = os.path.join("data", "policies", "shipment-guidelines-v2.md")
            if os.path.exists(shipment_policy_path):
                with open(shipment_policy_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract carrier information using regex
                    self._extract_carrier_info(content)
                    logger.info("Loaded shipment policy data")
            
            # Load payment rules
            payment_policy_path = os.path.join("data", "policies", "payout-rules-v3.md")
            if os.path.exists(payment_policy_path):
                with open(payment_policy_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract payment terms using regex
                    self._extract_payment_terms(content)
                    logger.info("Loaded payment policy data")
                    
        except Exception as e:
            logger.error(f"Error loading policy data: {e}")
    
    def _extract_carrier_info(self, content):
        """Extract carrier information from policy markdown"""
        # This is a simplified extraction - in production, you'd want more robust parsing
        lines = content.split('\n')
        current_mode = None
        
        for line in lines:
            if "International Air Freight" in line:
                current_mode = "air"
            elif "International Ocean Freight" in line:
                current_mode = "ocean"
            elif "Domestic Ground" in line:
                current_mode = "ground"
            elif line.startswith('- ') and current_mode:
                carrier_name = line.strip('- ').split('\n')[0]
                if carrier_name and current_mode in self.approved_carriers:
                    if carrier_name not in self.approved_carriers[current_mode]:
                        self.approved_carriers[current_mode].append(carrier_name)
    
    def _extract_payment_terms(self, content):
        """Extract payment terms from policy markdown"""
        # Extract supplier-specific terms using regex patterns
        supplier_sections = re.findall(r'### \d+\.\d+ (.+?)\n(.*?)(?=### |\Z)', content, re.DOTALL)
        
        for supplier_name, section in supplier_sections:
            if supplier_name in self.supplier_payment_terms:
                # Extract payment terms
                terms_match = re.search(r'Payment terms: (NET\d+)', section)
                discount_match = re.search(r'Early payment discount: (\d+(?:\.\d+)?)%', section)
                penalty_match = re.search(r'Late payment penalty: (\d+(?:\.\d+)?)%', section)
                
                if terms_match:
                    self.supplier_payment_terms[supplier_name]["terms"] = terms_match.group(1)
                if discount_match:
                    self.supplier_payment_terms[supplier_name]["early_discount"] = float(discount_match.group(1)) / 100
                if penalty_match:
                    self.supplier_payment_terms[supplier_name]["late_penalty"] = float(penalty_match.group(1)) / 100
    
    def detect_shipment_anomalies(self, shipment_data):
        """Detect anomalies in shipment data"""
        anomalies = []
        
        # Get shipment ID
        shipment_id = shipment_data.get("shipment_id")
        if not shipment_id:
            logger.warning("Shipment ID missing in data")
            return anomalies
        
        # Check for route deviation
        route_anomaly = self._check_route_deviation(shipment_data)
        if route_anomaly:
            anomalies.append(route_anomaly)
        
        # Check for timeline variation
        timeline_anomaly = self._check_timeline_variation(shipment_data)
        if timeline_anomaly:
            anomalies.append(timeline_anomaly)
        
        # Check for value discrepancy
        value_anomaly = self._check_value_discrepancy(shipment_data)
        if value_anomaly:
            anomalies.append(value_anomaly)
        
        # Check for duplicate shipment IDs
        duplicate_anomaly = self._check_duplicate_shipment(shipment_data)
        if duplicate_anomaly:
            anomalies.append(duplicate_anomaly)
        
        # Register the anomalies
        for anomaly in anomalies:
            self.anomalies.append(anomaly)
        
        return anomalies
    
    def detect_invoice_anomalies(self, invoice_data):
        """Detect anomalies in invoice data"""
        anomalies = []
        
        # Get invoice ID
        invoice_id = invoice_data.get("invoice_id")
        if not invoice_id:
            logger.warning("Invoice ID missing in data")
            return anomalies
        
        # Check for payment terms compliance
        terms_anomaly = self._check_payment_terms_compliance(invoice_data)
        if terms_anomaly:
            anomalies.append(terms_anomaly)
        
        # Check for amount discrepancy
        amount_anomaly = self._check_invoice_amount_discrepancy(invoice_data)
        if amount_anomaly:
            anomalies.append(amount_anomaly)
        
        # Check for approval workflow compliance
        approval_anomaly = self._check_approval_workflow_compliance(invoice_data)
        if approval_anomaly:
            anomalies.append(approval_anomaly)
        
        # Register the anomalies
        for anomaly in anomalies:
            self.anomalies.append(anomaly)
        
        return anomalies
    
    def _check_route_deviation(self, shipment_data):
        """Check for route deviation in shipment"""
        origin = shipment_data.get("origin")
        destination = shipment_data.get("destination")
        carrier = shipment_data.get("carrier")
        actual_route = shipment_data.get("actual_route")
        
        if not (origin and destination and carrier):
            return None
        
        # Create lookup key
        key = f"{origin}_{destination}_{carrier}"
        
        # Check if we have historical data for this route
        if key in self.historical_data["shipments"]:
            historical = self.historical_data["shipments"][key][0]  # Get first record
            expected_route = historical.get("expected_route")
            
            if expected_route and actual_route and expected_route != actual_route:
                # Calculate a simple risk score based on deviation
                risk_score = 0.8  # Default high risk
                
                # Create anomaly record
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                    "document_id": shipment_data.get("shipment_id"),
                    "anomaly_type": "route_deviation",
                    "description": f"Shipment route deviates from expected path. Expected: {expected_route}, Actual: {actual_route}",
                    "risk_score": risk_score,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "origin": origin,
                        "destination": destination,
                        "carrier": carrier,
                        "expected_route": expected_route,
                        "actual_route": actual_route
                    }
                }
        
        return None
    
    def _check_timeline_variation(self, shipment_data):
        """Check for timeline variation in shipment"""
        estimated_arrival = shipment_data.get("estimated_arrival")
        actual_arrival = shipment_data.get("actual_arrival")
        status = shipment_data.get("status")
        
        # If shipment is delivered and we have both dates
        if status == "Delayed" or (status == "Delivered" and estimated_arrival and actual_arrival):
            try:
                # Convert to datetime objects
                est_date = datetime.strptime(estimated_arrival, "%Y-%m-%d")
                
                if actual_arrival and actual_arrival != "null":
                    act_date = datetime.strptime(actual_arrival, "%Y-%m-%d")
                    
                    # Calculate delay in hours
                    delay_hours = (act_date - est_date).total_seconds() / 3600
                    
                    # Determine freight type (simplistic approach)
                    freight_type = "air" if delay_hours < 48 else "ocean"
                    
                    # Check if delay exceeds threshold
                    if delay_hours > self.timeline_variation_threshold[freight_type]:
                        # Calculate risk score
                        if freight_type == "air":
                            risk_score = min(0.9, 0.4 + (delay_hours / 72) * 0.5)
                        else:  # ocean
                            risk_score = min(0.9, 0.4 + (delay_hours / 168) * 0.5)
                        
                        # Create anomaly record
                        return {
                            "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                            "document_id": shipment_data.get("shipment_id"),
                            "anomaly_type": "timeline_delay",
                            "description": f"Shipment arrived {int(delay_hours)} hours later than estimated.",
                            "risk_score": risk_score,
                            "timestamp": int(datetime.now().timestamp()),
                            "metadata": {
                                "estimated_arrival": estimated_arrival,
                                "actual_arrival": actual_arrival,
                                "delay_hours": int(delay_hours),
                                "freight_type": freight_type
                            }
                        }
                elif status == "Delayed":
                    # For delayed shipments without actual arrival
                    current_date = datetime.now()
                    delay_hours = (current_date - est_date).total_seconds() / 3600
                    
                    # Determine freight type (simplistic approach)
                    freight_type = "air" if delay_hours < 48 else "ocean"
                    
                    # Check if delay exceeds threshold
                    if delay_hours > self.timeline_variation_threshold[freight_type]:
                        # Calculate risk score
                        if freight_type == "air":
                            risk_score = min(0.9, 0.4 + (delay_hours / 72) * 0.5)
                        else:  # ocean
                            risk_score = min(0.9, 0.4 + (delay_hours / 168) * 0.5)
                        
                        # Create anomaly record
                        return {
                            "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                            "document_id": shipment_data.get("shipment_id"),
                            "anomaly_type": "timeline_delay",
                            "description": f"Shipment is currently {int(delay_hours)} hours delayed.",
                            "risk_score": risk_score,
                            "timestamp": int(datetime.now().timestamp()),
                            "metadata": {
                                "estimated_arrival": estimated_arrival,
                                "current_delay_hours": int(delay_hours),
                                "freight_type": freight_type
                            }
                        }
            except Exception as e:
                logger.error(f"Error checking timeline variation: {e}")
        
        return None
    
    def _check_value_discrepancy(self, shipment_data):
        """Check for value discrepancy in shipment"""
        origin = shipment_data.get("origin")
        destination = shipment_data.get("destination")
        carrier = shipment_data.get("carrier")
        declared_value = shipment_data.get("value")
        
        if not (origin and destination and carrier and declared_value):
            return None
        
        try:
            # Convert value to float
            declared_value = float(declared_value)
            
            # Create lookup key
            key = f"{origin}_{destination}_{carrier}"
            
            # Check if we have historical data for this route
            if key in self.historical_data["shipments"]:
                historical = self.historical_data["shipments"][key][0]  # Get first record
                avg_value = historical.get("avg_value")
                std_value = historical.get("std_value")
                
                if avg_value:
                    # Calculate z-score
                    z_score = abs(declared_value - avg_value) / (std_value if std_value else avg_value * 0.1)
                    
                    # Calculate percentage difference
                    pct_diff = abs(declared_value - avg_value) / avg_value
                    
                    # Check if difference exceeds threshold
                    if pct_diff > self.value_discrepancy_threshold:
                        # Calculate risk score based on z-score
                        risk_score = min(0.95, 0.5 + min(z_score / 10, 0.45))
                        
                        # Create anomaly record
                        return {
                            "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                            "document_id": shipment_data.get("shipment_id"),
                            "anomaly_type": "value_discrepancy",
                            "description": f"Shipment value differs by {pct_diff:.1%} from historical average.",
                            "risk_score": risk_score,
                            "timestamp": int(datetime.now().timestamp()),
                            "metadata": {
                                "declared_value": declared_value,
                                "historical_avg_value": avg_value,
                                "percentage_difference": pct_diff,
                                "z_score": z_score
                            }
                        }
        except Exception as e:
            logger.error(f"Error checking value discrepancy: {e}")
        
        return None
    
    def _check_duplicate_shipment(self, shipment_data):
        """Check for duplicate shipment IDs"""
        shipment_id = shipment_data.get("shipment_id")
        
        # Simplistic approach - check if we've seen this shipment ID before
        # In a real system, this would check against a database
        for anomaly in self.anomalies:
            if (anomaly["document_id"] == shipment_id and 
                anomaly["anomaly_type"] != "duplicate_id"):
                
                # Create anomaly record
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                    "document_id": shipment_id,
                    "anomaly_type": "duplicate_id",
                    "description": f"Duplicate shipment ID detected: {shipment_id}",
                    "risk_score": 0.9,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "original_timestamp": anomaly["timestamp"]
                    }
                }
        
        return None
    
    def _check_payment_terms_compliance(self, invoice_data):
        """Check for payment terms compliance in invoice"""
        supplier = invoice_data.get("supplier")
        payment_terms = invoice_data.get("payment_terms")
        issue_date = invoice_data.get("issue_date")
        due_date = invoice_data.get("due_date")
        
        if not (supplier and payment_terms and issue_date and due_date):
            return None
        
        try:
            # Convert dates to datetime objects
            issue_dt = datetime.strptime(issue_date, "%Y-%m-%d")
            due_dt = datetime.strptime(due_date, "%Y-%m-%d")
            
            # Calculate days between issue and due dates
            days_diff = (due_dt - issue_dt).days
            
            # Extract the number from payment terms (e.g., "NET30" -> 30)
            expected_days = int(payment_terms.replace("NET", ""))
            
            # Check if due date is compliant with payment terms
            if abs(days_diff - expected_days) > 1:  # Allow 1 day tolerance
                # Calculate risk score
                risk_score = 0.7  # Default high risk for non-compliance
                
                # Create anomaly record
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                    "document_id": invoice_data.get("invoice_id"),
                    "anomaly_type": "payment_terms_noncompliance",
                    "description": f"Invoice due date doesn't match payment terms. Expected {expected_days} days, got {days_diff} days.",
                    "risk_score": risk_score,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "supplier": supplier,
                        "payment_terms": payment_terms,
                        "issue_date": issue_date,
                        "due_date": due_date,
                        "expected_days": expected_days,
                        "actual_days": days_diff
                    }
                }
        except Exception as e:
            logger.error(f"Error checking payment terms compliance: {e}")
        
        return None
    
    def _check_invoice_amount_discrepancy(self, invoice_data):
        """Check for amount discrepancy in invoice"""
        supplier = invoice_data.get("supplier")
        payment_terms = invoice_data.get("payment_terms")
        amount = invoice_data.get("amount")
        
        if not (supplier and payment_terms and amount):
            return None
        
        try:
            # Convert amount to float
            amount = float(amount)
            
            # Create lookup key
            key = f"{supplier}_{payment_terms}"
            
            # Check if we have historical data for this supplier
            if key in self.historical_data["invoices"]:
                historical = self.historical_data["invoices"][key][0]  # Get first record
                avg_amount = historical.get("avg_amount")
                std_amount = historical.get("std_amount")
                
                if avg_amount:
                    # Calculate z-score
                    z_score = abs(amount - avg_amount) / (std_amount if std_amount else avg_amount * 0.1)
                    
                    # Calculate percentage difference
                    pct_diff = abs(amount - avg_amount) / avg_amount
                    
                    # Check if difference exceeds threshold
                    if pct_diff > 0.25:  # 25% threshold for invoices
                        # Calculate risk score based on z-score
                        risk_score = min(0.9, 0.5 + min(z_score / 10, 0.4))
                        
                        # Create anomaly record
                        return {
                            "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                            "document_id": invoice_data.get("invoice_id"),
                            "anomaly_type": "invoice_amount_discrepancy",
                            "description": f"Invoice amount differs by {pct_diff:.1%} from historical average for this supplier.",
                            "risk_score": risk_score,
                            "timestamp": int(datetime.now().timestamp()),
                            "metadata": {
                                "amount": amount,
                                "historical_avg_amount": avg_amount,
                                "percentage_difference": pct_diff,
                                "z_score": z_score
                            }
                        }
        except Exception as e:
            logger.error(f"Error checking invoice amount discrepancy: {e}")
        
        return None
    
    def _check_approval_workflow_compliance(self, invoice_data):
        """Check for approval workflow compliance in invoice"""
        amount = invoice_data.get("amount")
        approver = invoice_data.get("approver")
        
        if not (amount and approver):
            return None
        
        try:
            # Convert amount to float
            amount = float(amount)
            
            # Determine required approval level
            required_approvers = []
            if amount < 5000:
                required_approvers = ["department_manager"]
            elif 5000 <= amount <= 25000:
                required_approvers = ["department_manager", "finance_director"]
            else:  # amount > 25000
                required_approvers = ["department_manager", "finance_director", "cfo"]
            
            # Check if approver matches required level
            # This is a simplified check - in reality would verify against a database
            if len(required_approvers) > 1 and not (
                "," in approver or  # Multiple approvers separated by comma
                "director" in approver.lower() or  # Finance director
                "cfo" in approver.lower()  # CFO
            ):
                # Calculate risk score
                risk_score = 0.8 if amount > 25000 else 0.6
                
                # Create anomaly record
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S")}",
                    "document_id": invoice_data.get("invoice_id"),
                    "anomaly_type": "approval_workflow_noncompliance",
                    "description": f"Invoice requires {len(required_approvers)} level(s) of approval but only has one approver.",
                    "risk_score": risk_score,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "amount": amount,
                        "current_approver": approver,
                        "required_approvers": required_approvers,
                        "approval_threshold": "$5,000-$25,000" if 5000 <= amount <= 25000 else ">$25,000"
                    }
                }
        except Exception as e:
            logger.error(f"Error checking approval workflow compliance: {e}")
        
        return None
    
    def get_anomalies(self, min_risk_score=0.5, start_date=None, end_date=None):
        """Get anomalies with filtering"""
        filtered_anomalies = []
        
        # Convert date strings to timestamps if provided
        start_ts = None
        end_ts = None
        
        if start_date:
            try:
                start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            except Exception as e:
                logger.error(f"Error parsing start date: {e}")
        
        if end_date:
            try:
                end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
                # Set end time to end of day
                end_ts += 86399  # seconds in a day - 1
            except Exception as e:
                logger.error(f"Error parsing end date: {e}")
        
        # Filter anomalies
        for anomaly in self.anomalies:
            # Check risk score
            if anomaly["risk_score"] < min_risk_score:
                continue
            
            # Check date range
            if start_ts and anomaly["timestamp"] < start_ts:
                continue
            
            if end_ts and anomaly["timestamp"] > end_ts:
                continue
            
            filtered_anomalies.append(anomaly)
        
        # Sort by timestamp (most recent first)
        filtered_anomalies.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return filtered_anomalies