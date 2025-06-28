
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from loguru import logger

class AnomalyDetector:
    """
    Utility class for detecting anomalies in logistics data
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.invoice_dir = f"{data_dir}/invoices"
        self.shipment_dir = f"{data_dir}/shipments"
        self.policy_dir = f"{data_dir}/policies"
        self.anomalies_dir = f"{data_dir}/anomalies"
        
        # Create directory for anomaly results
        os.makedirs(self.anomalies_dir, exist_ok=True)
        
        # Load any existing anomalies
        self.anomalies = self._load_existing_anomalies()
        
        # Configure thresholds
        self.thresholds = {
            "invoice_amount_deviation": 0.3,  # 30% deviation from historical average
            "payment_term_violation": 1.0,    # Any violation of payment terms
            "route_deviation_distance": 200,  # 200 km deviation from expected route
            "value_variance": 0.25,           # 25% deviation in declared value
            "delivery_time_deviation": 2.0    # 2 days deviation from expected delivery time
        }
    
    def _load_existing_anomalies(self) -> List[Dict[str, Any]]:
        """Load existing anomalies from file"""
        anomalies_file = f"{self.anomalies_dir}/anomalies.json"
        
        if not os.path.exists(anomalies_file):
            return []
        
        try:
            with open(anomalies_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading anomalies file: {e}")
            return []
    
    def _save_anomalies(self):
        """Save anomalies to file"""
        anomalies_file = f"{self.anomalies_dir}/anomalies.json"
        
        try:
            with open(anomalies_file, 'w') as f:
                json.dump(self.anomalies, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving anomalies file: {e}")
    
    def detect_invoice_anomalies(self, invoice_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in invoice data
        """
        anomalies = []
        
        # Extract key data
        invoice_id = invoice_data.get("invoice_id", "unknown")
        supplier = invoice_data.get("supplier", "unknown")
        amount = float(invoice_data.get("amount", 0.0))
        issue_date = invoice_data.get("issue_date", "unknown")
        due_date = invoice_data.get("due_date", "unknown")
        
        # Check for unusual amount
        supplier_history = self._get_supplier_invoice_history(supplier)
        if supplier_history:
            avg_amount = sum(inv.get("amount", 0.0) for inv in supplier_history) / len(supplier_history)
            deviation = abs(amount - avg_amount) / avg_amount if avg_amount > 0 else 0
            
            if deviation > self.thresholds["invoice_amount_deviation"]:
                anomalies.append({
                    "id": f"invoice_amount_{invoice_id}_{datetime.now().timestamp()}",
                    "document_id": invoice_id,
                    "anomaly_type": "invoice_amount_unusual",
                    "risk_score": min(1.0, deviation),
                    "description": f"Invoice amount (${amount:.2f}) deviates {deviation:.1%} from supplier average (${avg_amount:.2f})",
                    "timestamp": datetime.now().timestamp(),
                    "metadata": {
                        "supplier": supplier,
                        "amount": amount,
                        "avg_amount": avg_amount,
                        "deviation": deviation
                    }
                })
        
        # Check for payment term violations
        payment_policy = self._get_payment_policy()
        if payment_policy:
            # Check if supplier has special terms
            supplier_terms = next((terms for terms in payment_policy.get("supplier_terms", []) 
                                 if terms.get("supplier") == supplier), None)
            
            # Get standard terms if no supplier-specific terms
            std_terms = payment_policy.get("standard_terms", {})
            terms = supplier_terms or std_terms
            
            # Check payment terms
            if terms:
                std_due_days = terms.get("payment_due_days", 30)
                
                # Try to parse dates and calculate days difference
                try:
                    issue_dt = datetime.strptime(issue_date, "%Y-%m-%d")
                    due_dt = datetime.strptime(due_date, "%Y-%m-%d")
                    
                    actual_days = (due_dt - issue_dt).days
                    
                    if actual_days > std_due_days:
                        anomalies.append({
                            "id": f"payment_term_{invoice_id}_{datetime.now().timestamp()}",
                            "document_id": invoice_id,
                            "anomaly_type": "payment_term_violation",
                            "risk_score": 0.8,
                            "description": f"Payment terms violation: Due date is {actual_days} days from issue, but standard is {std_due_days} days",
                            "timestamp": datetime.now().timestamp(),
                            "metadata": {
                                "supplier": supplier,
                                "issue_date": issue_date,
                                "due_date": due_date,
                                "actual_days": actual_days,
                                "standard_days": std_due_days
                            }
                        })
                except (ValueError, TypeError):
                    # Skip date-based checks if dates are invalid
                    pass
        
        return anomalies
    
    def detect_shipment_anomalies(self, shipment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in shipment data
        """
        anomalies = []
        
        # Extract key data
        shipment_id = shipment_data.get("shipment_id", "unknown")
        origin = shipment_data.get("origin", "unknown")
        destination = shipment_data.get("destination", "unknown")
        departure_date = shipment_data.get("departure_date", "unknown")
        arrival_date = shipment_data.get("arrival_date", "unknown")
        carrier = shipment_data.get("carrier", "unknown")
        items = shipment_data.get("items", [])
        
        # Calculate total value
        total_value = sum(item.get("value", 0.0) for item in items)
        
        # Check for route deviation
        route_history = self._get_route_history(origin, destination)
        if route_history:
            # In a real implementation, we would use geospatial calculations
            # For now, we'll use a simple flag based on expected carrier
            expected_carriers = set(ship.get("carrier") for ship in route_history)
            
            if carrier not in expected_carriers:
                anomalies.append({
                    "id": f"route_carrier_{shipment_id}_{datetime.now().timestamp()}",
                    "document_id": shipment_id,
                    "anomaly_type": "route_carrier_unusual",
                    "risk_score": 0.7,
                    "description": f"Unusual carrier '{carrier}' for route {origin} to {destination}",
                    "timestamp": datetime.now().timestamp(),
                    "metadata": {
                        "origin": origin,
                        "destination": destination,
                        "carrier": carrier,
                        "expected_carriers": list(expected_carriers)
                    }
                })
        
        # Check for value variance
        if route_history:
            avg_value = sum(ship.get("total_value", 0.0) for ship in route_history) / len(route_history)
            deviation = abs(total_value - avg_value) / avg_value if avg_value > 0 else 0
            
            if deviation > self.thresholds["value_variance"]:
                anomalies.append({
                    "id": f"value_variance_{shipment_id}_{datetime.now().timestamp()}",
                    "document_id": shipment_id,
                    "anomaly_type": "shipment_value_unusual",
                    "risk_score": min(1.0, deviation),
                    "description": f"Shipment value (${total_value:.2f}) deviates {deviation:.1%} from route average (${avg_value:.2f})",
                    "timestamp": datetime.now().timestamp(),
                    "metadata": {
                        "origin": origin,
                        "destination": destination,
                        "total_value": total_value,
                        "avg_value": avg_value,
                        "deviation": deviation
                    }
                })
        
        # Check for delivery time anomaly
        try:
            departure_dt = datetime.strptime(departure_date, "%Y-%m-%d")
            arrival_dt = datetime.strptime(arrival_date, "%Y-%m-%d")
            
            actual_days = (arrival_dt - departure_dt).days
            
            if route_history:
                # Calculate average delivery time for this route
                delivery_times = []
                for ship in route_history:
                    try:
                        ship_dep = datetime.strptime(ship.get("departure_date", ""), "%Y-%m-%d")
                        ship_arr = datetime.strptime(ship.get("arrival_date", ""), "%Y-%m-%d")
                        delivery_times.append((ship_arr - ship_dep).days)
                    except (ValueError, TypeError):
                        pass
                
                if delivery_times:
                    avg_delivery_time = sum(delivery_times) / len(delivery_times)
                    deviation = abs(actual_days - avg_delivery_time)
                    
                    if deviation > self.thresholds["delivery_time_deviation"]:
                        anomalies.append({
                            "id": f"delivery_time_{shipment_id}_{datetime.now().timestamp()}",
                            "document_id": shipment_id,
                            "anomaly_type": "delivery_time_unusual",
                            "risk_score": min(0.9, deviation / 5),  # Scale score based on deviation
                            "description": f"Delivery time ({actual_days} days) deviates from average ({avg_delivery_time:.1f} days) for this route",
                            "timestamp": datetime.now().timestamp(),
                            "metadata": {
                                "origin": origin,
                                "destination": destination,
                                "departure_date": departure_date,
                                "arrival_date": arrival_date,
                                "actual_days": actual_days,
                                "avg_days": avg_delivery_time,
                                "deviation": deviation
                            }
                        })
        except (ValueError, TypeError):
            # Skip date-based checks if dates are invalid
            pass
        
        return anomalies
    
    def _get_supplier_invoice_history(self, supplier: str) -> List[Dict[str, Any]]:
        """
        Get historical invoice data for a supplier
        """
        # In a real implementation, this would query a database
        # For now, return placeholder data
        if supplier == "ABC Electronics":
            return [
                {"invoice_id": "INV001", "supplier": supplier, "amount": 5000.0},
                {"invoice_id": "INV002", "supplier": supplier, "amount": 4800.0},
                {"invoice_id": "INV003", "supplier": supplier, "amount": 5200.0}
            ]
        elif supplier == "XYZ Services":
            return [
                {"invoice_id": "INV101", "supplier": supplier, "amount": 2000.0},
                {"invoice_id": "INV102", "supplier": supplier, "amount": 1800.0}
            ]
        else:
            return []
    
    def _get_payment_policy(self) -> Dict[str, Any]:
        """
        Get current payment policy
        """
        # In a real implementation, this would parse the latest policy document
        # For now, return placeholder data
        return {
            "standard_terms": {
                "payment_due_days": 30,
                "early_payment_discount": 0.02,
                "early_payment_days": 10,
                "late_payment_fee": 0.05
            },
            "supplier_terms": [
                {
                    "supplier": "ABC Electronics",
                    "payment_due_days": 45,
                    "early_payment_discount": 0.01,
                    "early_payment_days": 15,
                    "late_payment_fee": 0.03
                },
                {
                    "supplier": "Premium Logistics",
                    "payment_due_days": 15,
                    "early_payment_discount": 0.03,
                    "early_payment_days": 5,
                    "late_payment_fee": 0.1
                }
            ]
        }
    
    def _get_route_history(self, origin: str, destination: str) -> List[Dict[str, Any]]:
        """
        Get historical shipment data for a route
        """
        # In a real implementation, this would query a database
        # For now, return placeholder data
        if origin == "New York, USA" and destination == "London, UK":
            return [
                {
                    "shipment_id": "SHP001",
                    "origin": origin,
                    "destination": destination,
                    "carrier": "Global Shipping Inc.",
                    "departure_date": "2023-01-10",
                    "arrival_date": "2023-01-17",
                    "total_value": 10000.0
                },
                {
                    "shipment_id": "SHP002",
                    "origin": origin,
                    "destination": destination,
                    "carrier": "Ocean Express",
                    "departure_date": "2023-02-15",
                    "arrival_date": "2023-02-21",
                    "total_value": 12000.0
                }
            ]
        elif origin == "Shanghai, China" and destination == "Hamburg, Germany":
            return [
                {
                    "shipment_id": "SHP101",
                    "origin": origin,
                    "destination": destination,
                    "carrier": "Asia Logistics",
                    "departure_date": "2023-01-05",
                    "arrival_date": "2023-01-25",
                    "total_value": 50000.0
                }
            ]
        else:
            return []
    
    def process_new_document(self, doc_path: str, doc_type: str, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a new document and detect anomalies
        """
        if doc_type == "invoice":
            anomalies = self.detect_invoice_anomalies(doc_data)
        elif doc_type == "shipment":
            anomalies = self.detect_shipment_anomalies(doc_data)
        else:
            # No anomaly detection for other document types
            return []
        
        # Save anomalies
        if anomalies:
            self.anomalies.extend(anomalies)
            self._save_anomalies()
            
            logger.info(f"Detected {len(anomalies)} anomalies in document {doc_path}")
        
        return anomalies
    
    def get_anomalies(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_risk_score: float = 0.5,
        doc_type: Optional[str] = None,
        anomaly_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get anomalies with optional filtering
        """
        filtered_anomalies = []
        
        for anomaly in self.anomalies:
            # Apply risk score filter
            if anomaly.get("risk_score", 0) < min_risk_score:
                continue
            
            # Apply document type filter
            if doc_type and anomaly.get("anomaly_type", "").split("_")[0] != doc_type:
                continue
            
            # Apply anomaly type filter
            if anomaly_type and anomaly.get("anomaly_type") != anomaly_type:
                continue
            
            # Apply date range filter
            anomaly_date = datetime.fromtimestamp(anomaly.get("timestamp", 0))
            
            if start_date:
                start = datetime.fromisoformat(start_date)
                if anomaly_date < start:
                    continue
            
            if end_date:
                end = datetime.fromisoformat(end_date)
                if anomaly_date > end:
                    continue
            
            filtered_anomalies.append(anomaly)
        
        return filtered_anomalies