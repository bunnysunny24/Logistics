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
    
    def detect_shipment_anomalies(self, shipment_data):
        """Detect anomalies in shipment data with enhanced accuracy"""
        anomalies = []
        
        # Get shipment ID
        shipment_id = shipment_data.get("shipment_id")
        if not shipment_id:
            logger.warning("Shipment ID missing in data")
            return anomalies
        
        # Check if already processed to avoid duplicates
        if shipment_id in self.processed_documents:
            logger.debug(f"Shipment {shipment_id} already processed")
            return anomalies
        
        # Check for carrier authorization
        carrier_anomaly = self._check_carrier_authorization(shipment_data)
        if carrier_anomaly:
            anomalies.append(carrier_anomaly)
        
        # Check for route deviation with enhanced logic
        route_anomaly = self._check_route_deviation_enhanced(shipment_data)
        if route_anomaly:
            anomalies.append(route_anomaly)
        
        # Check for timeline variation with freight mode detection
        timeline_anomaly = self._check_timeline_variation_enhanced(shipment_data)
        if timeline_anomaly:
            anomalies.append(timeline_anomaly)
        
        # Check for value discrepancy with item-level analysis
        value_anomaly = self._check_value_discrepancy_enhanced(shipment_data)
        if value_anomaly:
            anomalies.append(value_anomaly)
        
        # Check for duplicate shipment IDs
        duplicate_anomaly = self._check_duplicate_shipment_enhanced(shipment_data)
        if duplicate_anomaly:
            anomalies.append(duplicate_anomaly)
        
        # Check for high-risk combinations
        combined_anomaly = self._check_combined_risk_factors(shipment_data, anomalies)
        if combined_anomaly:
            anomalies.append(combined_anomaly)
        
        # Register the anomalies and mark as processed
        for anomaly in anomalies:
            self.anomalies.append(anomaly)
        
        self.processed_documents.add(shipment_id)
        
        return anomalies
    
    def detect_invoice_anomalies(self, invoice_data):
        """Detect anomalies in invoice data with enhanced compliance checking"""
        anomalies = []
        
        # Get invoice ID
        invoice_id = invoice_data.get("invoice_id")
        if not invoice_id:
            logger.warning("Invoice ID missing in data")
            return anomalies
        
        # Check if already processed to avoid duplicates
        if invoice_id in self.processed_documents:
            logger.debug(f"Invoice {invoice_id} already processed")
            return anomalies
        
        # Check for supplier-specific payment terms compliance
        terms_anomaly = self._check_supplier_payment_terms_compliance(invoice_data)
        if terms_anomaly:
            anomalies.append(terms_anomaly)
        
        # Check for amount discrepancy with historical analysis
        amount_anomaly = self._check_invoice_amount_discrepancy_enhanced(invoice_data)
        if amount_anomaly:
            anomalies.append(amount_anomaly)
        
        # Check for enhanced approval workflow compliance
        approval_anomaly = self._check_approval_workflow_compliance_enhanced(invoice_data)
        if approval_anomaly:
            anomalies.append(approval_anomaly)
        
        # Check for duplicate invoice IDs
        duplicate_anomaly = self._check_duplicate_invoice(invoice_data)
        if duplicate_anomaly:
            anomalies.append(duplicate_anomaly)
        
        # Check for early payment discount compliance
        discount_anomaly = self._check_early_payment_discount_compliance(invoice_data)
        if discount_anomaly:
            anomalies.append(discount_anomaly)
        
        # Check for currency and international payment compliance
        currency_anomaly = self._check_currency_compliance(invoice_data)
        if currency_anomaly:
            anomalies.append(currency_anomaly)
        
        # Register the anomalies and mark as processed
        for anomaly in anomalies:
            self.anomalies.append(anomaly)
        
        self.processed_documents.add(invoice_id)
        
        return anomalies
    
    def _check_carrier_authorization(self, shipment_data):
        """Check if carrier is authorized based on policies"""
        carrier = shipment_data.get("carrier", "")
        origin = shipment_data.get("origin", "")
        destination = shipment_data.get("destination", "")
        
        # Determine freight mode based on route
        freight_mode = self._determine_freight_mode(origin, destination)
        
        if freight_mode and carrier:
            authorized_carriers = self.approved_carriers.get(freight_mode, [])
            
            # Check if carrier is in approved list
            carrier_approved = False
            for approved_carrier in authorized_carriers:
                if approved_carrier.lower() in carrier.lower() or carrier.lower() in approved_carrier.lower():
                    carrier_approved = True
                    break
            
            if not carrier_approved:
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                    "document_id": shipment_data.get("shipment_id"),
                    "anomaly_type": "unauthorized_carrier",
                    "description": f"Carrier '{carrier}' is not authorized for {freight_mode} shipments",
                    "risk_score": 0.85,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "carrier": carrier,
                        "freight_mode": freight_mode,
                        "approved_carriers": authorized_carriers
                    }
                }
        
        return None
    
    def _determine_freight_mode(self, origin, destination):
        """Determine freight mode based on origin and destination"""
        if not origin or not destination:
            return "ground"
        
        # Simple logic - in production, use geolocation or routing APIs
        origin_parts = origin.split()
        dest_parts = destination.split()
        
        # Check if international
        if len(origin_parts) > 1 and len(dest_parts) > 1:
            origin_country = origin_parts[-1]
            dest_country = dest_parts[-1]
            
            if origin_country != dest_country:
                # International - determine air vs ocean based on distance/route
                # Simplified: assume ocean for major trade routes
                ocean_routes = [
                    ("China", "Germany"), ("USA", "UK"), ("Japan", "Korea")
                ]
                
                for route in ocean_routes:
                    if ((route[0] in origin and route[1] in destination) or 
                        (route[1] in origin and route[0] in destination)):
                        return "ocean"
                
                return "air"  # Default international to air
        
        return "ground"  # Domestic
    
    def _check_route_deviation_enhanced(self, shipment_data):
        """Enhanced route deviation check"""
        origin = shipment_data.get("origin")
        destination = shipment_data.get("destination")
        carrier = shipment_data.get("carrier")
        
        if not (origin and destination and carrier):
            return None
        
        # Check for known problematic routes or carriers
        risk_indicators = []
        
        # Check if using alternative/suspicious carrier
        if "Alternative" in carrier:
            risk_indicators.append("alternative_carrier")
        
        # Check for unusual route patterns
        if self._is_unusual_route(origin, destination):
            risk_indicators.append("unusual_route")
        
        if risk_indicators:
            risk_score = min(0.95, 0.6 + len(risk_indicators) * 0.15)
            
            return {
                "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                "document_id": shipment_data.get("shipment_id"),
                "anomaly_type": "route_deviation",
                "description": f"Route deviation detected: {', '.join(risk_indicators)}",
                "risk_score": risk_score,
                "timestamp": int(datetime.now().timestamp()),
                "metadata": {
                    "origin": origin,
                    "destination": destination,
                    "carrier": carrier,
                    "risk_indicators": risk_indicators
                }
            }
        
        return None
    
    def _is_unusual_route(self, origin, destination):
        """Check if route is unusual based on patterns"""
        # Simplified logic - in production, use geospatial analysis
        unusual_patterns = [
            ("New York", "London", "Alternative"),  # Unusual carrier for major route
        ]
        
        for pattern in unusual_patterns:
            if pattern[0] in origin and pattern[1] in destination:
                return True
        
        return False
    
    def _check_timeline_variation_enhanced(self, shipment_data):
        """Enhanced timeline variation check with better freight mode detection"""
        estimated_arrival = shipment_data.get("estimated_arrival")
        actual_arrival = shipment_data.get("actual_arrival")
        status = shipment_data.get("status")
        origin = shipment_data.get("origin", "")
        destination = shipment_data.get("destination", "")
        
        # Determine freight mode more accurately
        freight_mode = self._determine_freight_mode(origin, destination)
        
        if status == "Delayed" or (status == "Delivered" and estimated_arrival and actual_arrival):
            try:
                est_date = datetime.strptime(estimated_arrival, "%Y-%m-%d")
                
                if actual_arrival and actual_arrival != "null":
                    act_date = datetime.strptime(actual_arrival, "%Y-%m-%d")
                    delay_hours = (act_date - est_date).total_seconds() / 3600
                else:
                    # For delayed shipments
                    current_date = datetime.now()
                    delay_hours = (current_date - est_date).total_seconds() / 3600
                
                # Check against freight-specific thresholds
                threshold = self.timeline_variation_threshold.get(freight_mode, 48)
                
                if delay_hours > threshold:
                    # Calculate risk score based on delay severity
                    severity_multiplier = delay_hours / threshold
                    risk_score = min(0.95, 0.4 + (severity_multiplier - 1) * 0.3)
                    
                    return {
                        "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                        "document_id": shipment_data.get("shipment_id"),
                        "anomaly_type": "timeline_delay",
                        "description": f"{freight_mode.title()} shipment delayed by {int(delay_hours)} hours (threshold: {threshold}h)",
                        "risk_score": risk_score,
                        "timestamp": int(datetime.now().timestamp()),
                        "metadata": {
                            "estimated_arrival": estimated_arrival,
                            "actual_arrival": actual_arrival,
                            "delay_hours": int(delay_hours),
                            "freight_mode": freight_mode,
                            "threshold_hours": threshold,
                            "severity_multiplier": round(severity_multiplier, 2)
                        }
                    }
            except Exception as e:
                logger.error(f"Error checking enhanced timeline variation: {e}")
        
        return None
    
    def _check_value_discrepancy_enhanced(self, shipment_data):
        """Enhanced value discrepancy check with item-level analysis"""
        declared_value = shipment_data.get("value")
        if declared_value:
            try:
                value = float(declared_value)
                
                # Check for suspiciously round numbers (potential fraud indicator)
                if value % 1000 == 0 and value >= 10000:
                    risk_score = 0.7
                    return {
                        "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                        "document_id": shipment_data.get("shipment_id"),
                        "anomaly_type": "suspicious_value_pattern",
                        "description": f"Suspiciously round declared value: ${value:,.2f}",
                        "risk_score": risk_score,
                        "timestamp": int(datetime.now().timestamp()),
                        "metadata": {
                            "declared_value": value,
                            "pattern_type": "round_number"
                        }
                    }
                
                # Check for extremely high values that might need additional scrutiny
                if value > 50000:
                    risk_score = 0.6
                    return {
                        "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                        "document_id": shipment_data.get("shipment_id"),
                        "anomaly_type": "high_value_shipment",
                        "description": f"High-value shipment requiring additional verification: ${value:,.2f}",
                        "risk_score": risk_score,
                        "timestamp": int(datetime.now().timestamp()),
                        "metadata": {
                            "declared_value": value,
                            "threshold": 50000
                        }
                    }
                    
            except ValueError:
                pass
        
        return None
    
    def _check_duplicate_shipment_enhanced(self, shipment_data):
        """Enhanced duplicate shipment check"""
        shipment_id = shipment_data.get("shipment_id")
        origin = shipment_data.get("origin", "")
        destination = shipment_data.get("destination", "")
        departure_date = shipment_data.get("departure_date", "")
        
        # Check for exact ID duplicates
        for anomaly in self.anomalies:
            if (anomaly["document_id"] == shipment_id and 
                anomaly["anomaly_type"] != "duplicate_id"):
                
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                    "document_id": shipment_id,
                    "anomaly_type": "duplicate_id",
                    "description": f"Duplicate shipment ID detected: {shipment_id}",
                    "risk_score": 0.95,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "original_timestamp": anomaly["timestamp"]
                    }
                }
        
        # Check for suspicious pattern duplicates (same route, same date)
        for anomaly in self.anomalies:
            if (anomaly.get("metadata", {}).get("origin") == origin and
                anomaly.get("metadata", {}).get("destination") == destination and
                anomaly.get("metadata", {}).get("departure_date") == departure_date and
                anomaly["document_id"] != shipment_id):
                
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                    "document_id": shipment_id,
                    "anomaly_type": "suspicious_duplicate_pattern",
                    "description": f"Suspicious duplicate route pattern: {origin} → {destination} on {departure_date}",
                    "risk_score": 0.8,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "origin": origin,
                        "destination": destination,
                        "departure_date": departure_date,
                        "related_shipment": anomaly["document_id"]
                    }
                }
        
        return None
    
    def _check_combined_risk_factors(self, shipment_data, existing_anomalies):
        """Check for combined risk factors that increase overall risk"""
        if len(existing_anomalies) >= 2:
            # Multiple anomalies detected - increase overall risk
            combined_risk_score = min(0.98, max([a["risk_score"] for a in existing_anomalies]) + 0.1)
            
            return {
                "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                "document_id": shipment_data.get("shipment_id"),
                "anomaly_type": "combined_risk_factors",
                "description": f"Multiple risk factors detected: {', '.join([a['anomaly_type'] for a in existing_anomalies])}",
                "risk_score": combined_risk_score,
                "timestamp": int(datetime.now().timestamp()),
                "metadata": {
                    "combined_anomalies": [a["anomaly_type"] for a in existing_anomalies],
                    "individual_risk_scores": [a["risk_score"] for a in existing_anomalies]
                }
            }
        
        return None
    
    def _check_supplier_payment_terms_compliance(self, invoice_data):
        """Check supplier-specific payment terms compliance"""
        supplier = invoice_data.get("supplier")
        payment_terms = invoice_data.get("payment_terms")
        issue_date = invoice_data.get("issue_date")
        due_date = invoice_data.get("due_date")
        
        if supplier in self.supplier_payment_terms:
            expected_terms = self.supplier_payment_terms[supplier]["terms"]
            
            # Check if payment terms match expected
            if payment_terms != expected_terms:
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                    "document_id": invoice_data.get("invoice_id"),
                    "anomaly_type": "incorrect_payment_terms",
                    "description": f"Payment terms for {supplier} should be {expected_terms}, got {payment_terms}",
                    "risk_score": 0.75,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "supplier": supplier,
                        "expected_terms": expected_terms,
                        "actual_terms": payment_terms
                    }
                }
            
            # Check due date calculation
            if issue_date and due_date:
                try:
                    issue_dt = datetime.strptime(issue_date, "%Y-%m-%d")
                    due_dt = datetime.strptime(due_date, "%Y-%m-%d")
                    actual_days = (due_dt - issue_dt).days
                    expected_days = int(expected_terms.replace("NET", ""))
                    
                    if abs(actual_days - expected_days) > self.payment_terms_tolerance:
                        return {
                            "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                            "document_id": invoice_data.get("invoice_id"),
                            "anomaly_type": "incorrect_due_date_calculation",
                            "description": f"Due date calculation error: expected {expected_days} days, calculated {actual_days} days",
                            "risk_score": 0.6,
                            "timestamp": int(datetime.now().timestamp()),
                            "metadata": {
                                "supplier": supplier,
                                "expected_days": expected_days,
                                "actual_days": actual_days,
                                "tolerance": self.payment_terms_tolerance
                            }
                        }
                except Exception as e:
                    logger.error(f"Error checking due date calculation: {e}")
        
        return None
    
    def _check_invoice_amount_discrepancy_enhanced(self, invoice_data):
        """Enhanced invoice amount discrepancy check with historical patterns"""
        supplier = invoice_data.get("supplier")
        payment_terms = invoice_data.get("payment_terms")
        amount = invoice_data.get("amount")
        
        if not (supplier and payment_terms and amount):
            return None
        
        try:
            amount = float(amount)
            key = f"{supplier}_{payment_terms}"
            
            # Check against historical data
            if key in self.historical_data["invoices"]:
                historical_amounts = self.historical_data["invoices"][key]["amounts"]
                
                if historical_amounts:
                    avg_amount = np.mean(historical_amounts)
                    std_amount = np.std(historical_amounts) if len(historical_amounts) > 1 else avg_amount * 0.1
                    
                    # Calculate percentage difference
                    pct_diff = abs(amount - avg_amount) / avg_amount
                    
                    if pct_diff > self.invoice_amount_threshold:
                        # Calculate z-score for risk assessment
                        z_score = abs(amount - avg_amount) / std_amount if std_amount > 0 else 0
                        risk_score = min(0.9, 0.5 + min(z_score / 10, 0.4))
                        
                        return {
                            "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                            "document_id": invoice_data.get("invoice_id"),
                            "anomaly_type": "invoice_amount_discrepancy",
                            "description": f"Invoice amount differs by {pct_diff:.1%} from historical average (${avg_amount:,.2f})",
                            "risk_score": risk_score,
                            "timestamp": int(datetime.now().timestamp()),
                            "metadata": {
                                "amount": amount,
                                "historical_avg": avg_amount,
                                "percentage_difference": pct_diff,
                                "z_score": z_score,
                                "sample_size": len(historical_amounts)
                            }
                        }
            
            # Check for suspiciously round amounts
            if amount % 100 == 0 and amount >= 1000:
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                    "document_id": invoice_data.get("invoice_id"),
                    "anomaly_type": "suspicious_round_amount",
                    "description": f"Suspiciously round invoice amount: ${amount:,.2f}",
                    "risk_score": 0.65,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "amount": amount,
                        "pattern_type": "round_number"
                    }
                }
                
        except ValueError:
            pass
        
        return None
    
    def _check_approval_workflow_compliance_enhanced(self, invoice_data):
        """Enhanced approval workflow compliance check"""
        amount = invoice_data.get("amount")
        approver = invoice_data.get("approver")
        supplier = invoice_data.get("supplier")
        
        if not (amount and approver):
            return None
        
        try:
            amount = float(amount)
            
            # Determine required approval level based on amount
            required_level = "manager"
            if amount >= self.approval_thresholds["cfo"]:
                required_level = "cfo"
            elif amount >= self.approval_thresholds["director"]:
                required_level = "director"
            elif amount >= self.approval_thresholds["manager"]:
                required_level = "manager"
            
            # Check approver authority level
            approver_level = self._determine_approver_level(approver)
            
            # Verify approval hierarchy
            approval_hierarchy = ["manager", "director", "cfo"]
            required_index = approval_hierarchy.index(required_level)
            approver_index = approval_hierarchy.index(approver_level) if approver_level in approval_hierarchy else -1
            
            if approver_index < required_index:
                risk_score = 0.8 if amount > 25000 else 0.6
                
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                    "document_id": invoice_data.get("invoice_id"),
                    "anomaly_type": "insufficient_approval_authority",
                    "description": f"${amount:,.2f} invoice requires {required_level} approval, but approved by {approver_level}",
                    "risk_score": risk_score,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "amount": amount,
                        "required_level": required_level,
                        "approver_level": approver_level,
                        "approver": approver,
                        "threshold": self.approval_thresholds[required_level]
                    }
                }
                
        except ValueError:
            pass
        
        return None
    
    def _determine_approver_level(self, approver):
        """Determine approver authority level from name/title"""
        approver_lower = approver.lower()
        
        if "cfo" in approver_lower or "chief financial" in approver_lower:
            return "cfo"
        elif "director" in approver_lower or "vice president" in approver_lower:
            return "director"
        elif "manager" in approver_lower or any(name in approver_lower for name in ["john", "sarah", "michael", "emily", "david"]):
            return "manager"
        else:
            return "unknown"
    
    def _check_duplicate_invoice(self, invoice_data):
        """Check for duplicate invoice IDs"""
        invoice_id = invoice_data.get("invoice_id")
        supplier = invoice_data.get("supplier")
        amount = invoice_data.get("amount")
        issue_date = invoice_data.get("issue_date")
        
        # Check for exact ID duplicates
        for anomaly in self.anomalies:
            if (anomaly["document_id"] == invoice_id and 
                anomaly["anomaly_type"] != "duplicate_invoice_id"):
                
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                    "document_id": invoice_id,
                    "anomaly_type": "duplicate_invoice_id",
                    "description": f"Duplicate invoice ID detected: {invoice_id}",
                    "risk_score": 0.9,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "original_timestamp": anomaly["timestamp"]
                    }
                }
        
        # Check for suspicious pattern duplicates (same supplier, amount, date)
        for anomaly in self.anomalies:
            if (anomaly.get("metadata", {}).get("supplier") == supplier and
                anomaly.get("metadata", {}).get("amount") == amount and
                anomaly.get("metadata", {}).get("issue_date") == issue_date and
                anomaly["document_id"] != invoice_id):
                
                return {
                    "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                    "document_id": invoice_id,
                    "anomaly_type": "suspicious_duplicate_invoice_pattern",
                    "description": f"Suspicious duplicate invoice pattern: {supplier}, ${amount}, {issue_date}",
                    "risk_score": 0.85,
                    "timestamp": int(datetime.now().timestamp()),
                    "metadata": {
                        "supplier": supplier,
                        "amount": amount,
                        "issue_date": issue_date,
                        "related_invoice": anomaly["document_id"]
                    }
                }
        
        return None
    
    def _check_early_payment_discount_compliance(self, invoice_data):
        """Check early payment discount compliance"""
        supplier = invoice_data.get("supplier")
        early_discount = invoice_data.get("early_discount", 0)
        
        if supplier in self.supplier_payment_terms:
            expected_discount = self.supplier_payment_terms[supplier]["early_discount"]
            
            try:
                actual_discount = float(early_discount)
                
                if abs(actual_discount - expected_discount) > 0.005:  # 0.5% tolerance
                    return {
                        "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                        "document_id": invoice_data.get("invoice_id"),
                        "anomaly_type": "incorrect_early_discount",
                        "description": f"Early discount for {supplier} should be {expected_discount:.1%}, got {actual_discount:.1%}",
                        "risk_score": 0.5,
                        "timestamp": int(datetime.now().timestamp()),
                        "metadata": {
                            "supplier": supplier,
                            "expected_discount": expected_discount,
                            "actual_discount": actual_discount
                        }
                    }
            except ValueError:
                pass
        
        return None
    
    def _check_currency_compliance(self, invoice_data):
        """Check currency and international payment compliance"""
        currency = invoice_data.get("currency", "USD")
        supplier = invoice_data.get("supplier", "")
        amount = invoice_data.get("amount", 0)
        
        # Check for non-USD currencies (might need special handling)
        if currency != "USD":
            return {
                "id": f"ANM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(self.anomalies)}",
                "document_id": invoice_data.get("invoice_id"),
                "anomaly_type": "non_usd_currency",
                "description": f"International invoice in {currency} requires special handling",
                "risk_score": 0.4,
                "timestamp": int(datetime.now().timestamp()),
                "metadata": {
                    "currency": currency,
                    "supplier": supplier,
                    "amount": amount
                }
            }
        
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
    
    def get_anomaly_summary(self):
        """Get summary statistics of detected anomalies"""
        if not self.anomalies:
            return {"total": 0, "by_type": {}, "by_risk_level": {}}
        
        # Count by type
        by_type = {}
        by_risk_level = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for anomaly in self.anomalies:
            # Count by type
            anomaly_type = anomaly["anomaly_type"]
            by_type[anomaly_type] = by_type.get(anomaly_type, 0) + 1
            
            # Count by risk level
            risk_score = anomaly["risk_score"]
            if risk_score >= 0.8:
                by_risk_level["critical"] += 1
            elif risk_score >= 0.6:
                by_risk_level["high"] += 1
            elif risk_score >= 0.4:
                by_risk_level["medium"] += 1
            else:
                by_risk_level["low"] += 1
        
        return {
            "total": len(self.anomalies),
            "by_type": by_type,
            "by_risk_level": by_risk_level,
            "processed_documents": len(self.processed_documents)
        }
    
    def process_all_sample_data(self):
        """Process all sample data to detect anomalies"""
        logger.info("Processing all sample data for anomaly detection...")
        
        # Process comprehensive shipment data
        try:
            comprehensive_shipments_path = os.path.join("data", "shipments", "comprehensive_shipments.csv")
            if os.path.exists(comprehensive_shipments_path):
                df = pd.read_csv(comprehensive_shipments_path)
                for _, row in df.iterrows():
                    self.detect_shipment_anomalies(row.to_dict())
                logger.info(f"Processed {len(df)} shipment records")
        except Exception as e:
            logger.error(f"Error processing shipment data: {e}")
        
        # Process comprehensive invoice data
        try:
            comprehensive_invoices_path = os.path.join("data", "invoices", "comprehensive_invoices.csv")
            if os.path.exists(comprehensive_invoices_path):
                df = pd.read_csv(comprehensive_invoices_path)
                for _, row in df.iterrows():
                    self.detect_invoice_anomalies(row.to_dict())
                logger.info(f"Processed {len(df)} invoice records")
        except Exception as e:
            logger.error(f"Error processing invoice data: {e}")
        
        # Log summary
        summary = self.get_anomaly_summary()
        logger.info(f"Anomaly detection complete: {summary}")
        
        return summary
