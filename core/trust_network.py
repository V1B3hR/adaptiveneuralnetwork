"""
Enhanced Trust Network System

Provides advanced trust management with:
- Suspicion detection based on behavioral patterns
- Community verification for suspicious nodes
- Manipulation pattern detection (love bombing, push-pull)
- Configurable trust thresholds and volatility limits
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict


class TrustNetwork:
    def __init__(self, node_id):
        self.node_id = node_id
        self.trust_network = {}
        self.interaction_history = {}  # Track patterns over time
        self.suspicion_alerts = {}  # Track suspicious behaviors
        
        # Configurable thresholds
        self.SUSPICION_THRESHOLD = 0.3  # When to start community verification
        self.PARANOIA_THRESHOLD = 0.1   # Too low - we're being paranoid
        self.TRUST_VOLATILITY_LIMIT = 0.2  # Max trust change per interaction
        
    def update_trust(self, target, signal_type, context=None):
        """Update trust with suspicion detection and community verification"""
        current_trust = self.trust_network.get(target.node_id, 0.5)
        
        # Track interaction patterns
        self._record_interaction(target.node_id, signal_type, context)
        
        # Calculate trust delta based on signal type
        trust_delta = self._calculate_trust_delta(signal_type, target, context)
        
        # Apply volatility limits to prevent sudden swings
        trust_delta = max(-self.TRUST_VOLATILITY_LIMIT, 
                         min(self.TRUST_VOLATILITY_LIMIT, trust_delta))
        
        new_trust = current_trust + trust_delta
        new_trust = max(0.0, min(1.0, new_trust))  # Keep in [0, 1]
        
        # Check for suspicious patterns
        if self._detect_suspicious_pattern(target.node_id, new_trust, current_trust):
            return self._initiate_community_verification(target, signal_type, context)
        
        # Update trust
        self.trust_network[target.node_id] = new_trust
        
        # Update interaction history with final trust value
        if target.node_id in self.interaction_history and self.interaction_history[target.node_id]:
            self.interaction_history[target.node_id][-1]['trust_after'] = new_trust
        
        # Clear suspicion if trust is recovering naturally
        if target.node_id in self.suspicion_alerts and new_trust > self.SUSPICION_THRESHOLD:
            self.suspicion_alerts[target.node_id]['status'] = 'resolved'
        
        return new_trust
    
    def _calculate_trust_delta(self, signal_type, target, context):
        """Calculate trust change based on signal type and context"""
        base_deltas = {
            # Positive signals
            "resource": 0.1,          # Sharing resources
            "joy_share": 0.04,        # Sharing positive emotions
            "grief_support_request": 0.03,  # Vulnerability
            "celebration_invite": 0.03,      # Inclusion
            "comfort_request": 0.02,         # Reaching out
            
            # Neutral signals
            "information": 0.01,      # Sharing info
            "memory": 0.05,           # Sharing memories
            "acknowledgment": 0.0,    # Just acknowledging
            
            # Negative signals
            "resource_denial": -0.05, # Refusing to share when able
            "exclusion": -0.03,       # Excluding from group activities
            "gossip": -0.04,          # Spreading rumors
            "betrayal": -0.15,        # Breaking explicit trust
            "deception": -0.2,        # Caught lying
        }
        
        delta = base_deltas.get(signal_type, 0.0)
        
        # Modify based on context
        if context:
            # Repeated positive behavior builds trust faster
            if context.get('consistent_positive', False):
                delta *= 1.2
            
            # But too much too fast is suspicious
            if context.get('sudden_change', False):
                delta *= 0.5  # Dampen the change
                
            # Consider reciprocity
            if context.get('reciprocated', False):
                delta *= 1.1
                
        return delta
    
    def _detect_suspicious_pattern(self, node_id, new_trust, old_trust):
        """Detect patterns that warrant community verification"""
        
        # Check if trust is dropping below threshold
        if new_trust < self.SUSPICION_THRESHOLD and old_trust >= self.SUSPICION_THRESHOLD:
            return True
        
        # Check for erratic behavior (trust bouncing up and down)
        if node_id in self.interaction_history:
            recent = self.interaction_history[node_id][-10:]  # Last 10 interactions
            if len(recent) >= 5:
                trust_changes = [h['trust_after'] - h['trust_before'] for h in recent 
                               if h['trust_after'] is not None]
                if len(trust_changes) > 1:
                    volatility = np.std(trust_changes)
                    
                    if volatility > 0.15:  # High volatility is suspicious
                        return True
        
        # Check for manipulation patterns
        if self._detect_manipulation_pattern(node_id):
            return True
            
        return False
    
    def _detect_manipulation_pattern(self, node_id):
        """Detect potential manipulation tactics"""
        if node_id not in self.interaction_history:
            return False
            
        recent = self.interaction_history[node_id][-20:]
        
        # Love bombing pattern: too many positive signals too quickly
        positive_signals = ['resource', 'joy_share', 'celebration_invite', 'comfort_request']
        positive_count = sum(1 for h in recent[-5:] 
                           if h['signal_type'] in positive_signals)
        if positive_count >= 4:
            return True
        
        # Push-pull pattern: alternating positive and negative
        if len(recent) >= 6:
            pattern = []
            for h in recent[-6:]:
                if h['trust_after'] is not None and h['trust_before'] is not None:
                    pattern.append(h['trust_after'] > h['trust_before'])
                    
            if len(pattern) >= 6:
                if pattern == [True, False, True, False, True, False] or \
                   pattern == [False, True, False, True, False, True]:
                    return True
        
        return False
    
    def _initiate_community_verification(self, target, signal_type, context):
        """Start a community verification process for suspicious behavior"""
        
        # Record the suspicion
        self.suspicion_alerts[target.node_id] = {
            'timestamp': context.get('timestamp', 'now') if context else 'now',
            'signal_type': signal_type,
            'status': 'pending_verification',
            'trust_level': self.trust_network.get(target.node_id, 0.5)
        }
        
        # Create verification request for community
        verification_request = {
            'type': 'trust_verification',
            'subject': target.node_id,
            'requester': self.node_id,
            'reason': self._generate_suspicion_reason(target.node_id, signal_type),
            'recent_interactions': self.interaction_history.get(target.node_id, [])[-5:],
            'current_trust': self.trust_network.get(target.node_id, 0.5)
        }
        
        # This would broadcast to trusted neighbors for their opinion
        self._broadcast_to_trusted_neighbors(verification_request)
        
        # Return current trust (frozen until verification)
        return self.trust_network.get(target.node_id, 0.5)
    
    def _generate_suspicion_reason(self, node_id, signal_type):
        """Generate human-readable reason for suspicion"""
        reasons = []
        
        if self.trust_network.get(node_id, 0.5) < self.SUSPICION_THRESHOLD:
            reasons.append(f"Trust level dropped below threshold ({self.SUSPICION_THRESHOLD})")
        
        if node_id in self.interaction_history:
            recent = self.interaction_history[node_id][-10:]
            trust_changes = [h['trust_after'] - h['trust_before'] for h in recent 
                           if h['trust_after'] is not None and h['trust_before'] is not None]
            if len(trust_changes) > 1:
                volatility = np.std(trust_changes)
                if volatility > 0.15:
                    reasons.append("Erratic behavior pattern detected")
        
        if self._detect_manipulation_pattern(node_id):
            reasons.append("Potential manipulation tactics observed")
            
        return "; ".join(reasons) if reasons else "General suspicion"
    
    def _broadcast_to_trusted_neighbors(self, verification_request):
        """Send verification request to trusted community members"""
        trusted_neighbors = [
            node_id for node_id, trust in self.trust_network.items()
            if trust > 0.6  # Only ask those we trust
        ]
        
        for neighbor in trusted_neighbors[:5]:  # Ask up to 5 trusted neighbors
            # This would actually send the request
            # For now, we'll just log it
            print(f"Requesting trust verification from {neighbor} about {verification_request['subject']}")
    
    def process_community_feedback(self, subject_id, feedback_list):
        """Process community feedback on suspicious node"""
        if not feedback_list:
            return
        
        # Aggregate community opinion
        trust_votes = [f['trust_level'] for f in feedback_list]
        avg_community_trust = np.mean(trust_votes)
        trust_std = np.std(trust_votes) if len(trust_votes) > 1 else 0
        
        # High agreement in community
        if trust_std < 0.15:
            # Adjust our trust toward community consensus
            current = self.trust_network.get(subject_id, 0.5)
            adjusted = current * 0.3 + avg_community_trust * 0.7
            self.trust_network[subject_id] = adjusted
            
            # Update suspicion status
            if subject_id in self.suspicion_alerts:
                if adjusted > self.SUSPICION_THRESHOLD:
                    self.suspicion_alerts[subject_id]['status'] = 'cleared_by_community'
                else:
                    self.suspicion_alerts[subject_id]['status'] = 'confirmed_suspicious'
        else:
            # Community is divided - maintain cautious stance
            if subject_id in self.suspicion_alerts:
                self.suspicion_alerts[subject_id]['status'] = 'community_divided'
            # Maybe slightly decrease trust due to uncertainty
            if subject_id in self.trust_network:
                self.trust_network[subject_id] *= 0.95
    
    def _record_interaction(self, node_id, signal_type, context):
        """Keep history of interactions for pattern detection"""
        if node_id not in self.interaction_history:
            self.interaction_history[node_id] = []
        
        self.interaction_history[node_id].append({
            'signal_type': signal_type,
            'trust_before': self.trust_network.get(node_id, 0.5),
            'trust_after': None,  # Will be filled after trust update
            'timestamp': context.get('timestamp', 'now') if context else 'now',
            'context': context
        })
        
        # Keep history manageable
        if len(self.interaction_history[node_id]) > 100:
            self.interaction_history[node_id] = self.interaction_history[node_id][-50:]
    
    def get_trust_summary(self):
        """Get overview of trust network health"""
        if not self.trust_network:
            return {
                'average_trust': 0.0,
                'trusted_nodes': 0,
                'suspicious_nodes': 0,
                'active_alerts': 0,
                'paranoia_warning': False
            }
        
        trust_values = list(self.trust_network.values())
        avg_trust = np.mean(trust_values)
        suspicious_count = sum(1 for t in trust_values if t < self.SUSPICION_THRESHOLD)
        paranoid_check = sum(1 for t in trust_values if t < self.PARANOIA_THRESHOLD)
        
        summary = {
            'average_trust': avg_trust,
            'trusted_nodes': sum(1 for t in trust_values if t > 0.6),
            'suspicious_nodes': suspicious_count,
            'active_alerts': sum(1 for a in self.suspicion_alerts.values() if a['status'] == 'pending_verification'),
            'paranoia_warning': paranoid_check > len(trust_values) * 0.3  # If >30% are below paranoia threshold
        }
        
        return summary
    
    def get_trust(self, node_id):
        """Get trust level for a specific node"""
        return self.trust_network.get(node_id, 0.5)
    
    def set_trust(self, node_id, trust_value):
        """Set trust level for a specific node (for initialization)"""
        self.trust_network[node_id] = max(0.0, min(1.0, trust_value))