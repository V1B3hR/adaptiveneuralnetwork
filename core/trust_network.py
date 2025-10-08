"""
Enhanced Trust Network System

Provides advanced trust management with:
- Suspicion detection based on behavioral patterns
- Community verification for suspicious nodes
- Manipulation pattern detection (love bombing, push-pull)
- Configurable trust thresholds and volatility limits
"""


import numpy as np


class TrustNetwork:
    def __init__(self, node_id):
        self.node_id = node_id
        self.trust_network = {}
        self.interaction_history = {}  # Track patterns over time
        self.suspicion_alerts = {}  # Track suspicious behaviors
        self.last_decay_time = {}  # Track when trust was last decayed for each node

        # Configurable thresholds
        self.SUSPICION_THRESHOLD = 0.25  # Lowered from 0.3 for more aggressive detection
        self.PARANOIA_THRESHOLD = 0.1   # Too low - we're being paranoid
        self.TRUST_VOLATILITY_LIMIT = 0.2  # Max trust change per interaction

        # Trust decay/recovery parameters
        self.TRUST_DECAY_RATE = 0.01  # Trust decay per time unit without interaction
        self.TRUST_RECOVERY_RATE = 0.02  # Trust recovery rate for positive interactions
        self.MIN_TRUST_THRESHOLD = 0.1  # Minimum trust level to maintain
        self.DECAY_TIME_THRESHOLD = 10  # Time units before trust starts decaying

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

        # Record interaction time for decay tracking
        self.last_decay_time[target.node_id] = self._get_current_time()

        # Apply trust recovery for positive interactions
        if trust_delta > 0:
            recovery_factor = min(trust_delta / self.TRUST_VOLATILITY_LIMIT, 1.0)
            self.apply_trust_recovery(target.node_id, recovery_factor)

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

        # Love bombing pattern: too many positive signals too quickly (lowered threshold)
        positive_signals = ['resource', 'joy_share', 'celebration_invite', 'comfort_request']
        positive_count = sum(1 for h in recent[-5:]
                           if h['signal_type'] in positive_signals)
        if positive_count >= 3:  # Lowered from 4 to 3 for more sensitive detection
            return True

        # Rapid trust building pattern (new detection)
        if len(recent) >= 3:
            trust_increases = []
            for h in recent[-3:]:
                if h['trust_after'] is not None and h['trust_before'] is not None:
                    if h['trust_after'] > h['trust_before']:
                        trust_increases.append(h['trust_after'] - h['trust_before'])

            # If all recent interactions are trust increases above threshold
            if len(trust_increases) >= 3 and all(increase > 0.1 for increase in trust_increases):
                return True

        # Push-pull pattern: alternating positive and negative (enhanced)
        if len(recent) >= 4:  # Reduced from 6 to 4 for faster detection
            pattern = []
            for h in recent[-4:]:
                if h['trust_after'] is not None and h['trust_before'] is not None:
                    pattern.append(h['trust_after'] > h['trust_before'])

            if len(pattern) >= 4:
                # Look for alternating patterns
                alternating_patterns = [
                    [True, False, True, False],
                    [False, True, False, True],
                    [True, False, False, True],  # Additional patterns
                    [False, True, True, False]
                ]
                if pattern in alternating_patterns:
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
            # print(f"Requesting trust verification from {neighbor} about {verification_request['subject']}")  # Silenced for cleaner output
            pass

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
        self.last_decay_time[node_id] = self._get_current_time()

    def apply_trust_decay(self, current_time=None):
        """Apply trust decay to nodes that haven't interacted recently"""
        if current_time is None:
            current_time = self._get_current_time()

        nodes_to_decay = []
        for node_id, trust_level in self.trust_network.items():
            last_interaction = self.last_decay_time.get(node_id, current_time)
            time_since_interaction = current_time - last_interaction

            # Only decay if enough time has passed
            if time_since_interaction >= self.DECAY_TIME_THRESHOLD:
                decay_amount = self.TRUST_DECAY_RATE * (time_since_interaction - self.DECAY_TIME_THRESHOLD)
                new_trust = max(self.MIN_TRUST_THRESHOLD, trust_level - decay_amount)

                if new_trust != trust_level:
                    nodes_to_decay.append((node_id, new_trust))

        # Apply decays
        for node_id, new_trust in nodes_to_decay:
            self.trust_network[node_id] = new_trust

        return len(nodes_to_decay)

    def apply_trust_recovery(self, node_id, recovery_factor=1.0):
        """Apply trust recovery for positive interactions"""
        if node_id not in self.trust_network:
            return

        current_trust = self.trust_network[node_id]
        recovery_amount = self.TRUST_RECOVERY_RATE * recovery_factor

        # Recovery is stronger for lower trust values (easier to recover from bottom)
        trust_recovery_multiplier = (1.0 - current_trust) + 0.5
        recovery_amount *= trust_recovery_multiplier

        new_trust = min(1.0, current_trust + recovery_amount)
        self.trust_network[node_id] = new_trust
        self.last_decay_time[node_id] = self._get_current_time()

        return new_trust - current_trust

    def _get_current_time(self):
        """Get current time (can be overridden for testing)"""
        import time
        return time.time()

    # Advanced Trust Network Visualization and Monitoring

    def generate_trust_network_graph(self):
        """Generate a graph representation of the trust network for visualization"""
        graph_data = {
            'nodes': [],
            'edges': [],
            'metadata': self.get_trust_summary()
        }

        # Add current node
        graph_data['nodes'].append({
            'id': self.node_id,
            'label': f'Node {self.node_id}',
            'type': 'self',
            'trust_level': 1.0,  # Self-trust
            'status': 'active'
        })

        # Add connected nodes
        for node_id, trust_level in self.trust_network.items():
            status = 'suspicious' if trust_level < self.SUSPICION_THRESHOLD else 'trusted'
            if node_id in self.suspicion_alerts:
                status = 'alert'

            graph_data['nodes'].append({
                'id': node_id,
                'label': f'Node {node_id}',
                'type': 'peer',
                'trust_level': trust_level,
                'status': status
            })

            # Add edge from self to peer
            edge_color = 'red' if trust_level < self.SUSPICION_THRESHOLD else 'green'
            graph_data['edges'].append({
                'source': self.node_id,
                'target': node_id,
                'weight': trust_level,
                'color': edge_color,
                'thickness': max(1, int(trust_level * 5))
            })

        return graph_data

    def get_trust_network_metrics(self):
        """Get comprehensive trust network health metrics"""
        if not self.trust_network:
            return {
                'total_connections': 0,
                'average_trust': 0,
                'trust_variance': 0,
                'network_resilience': 0,
                'suspicious_ratio': 0,
                'alert_count': 0
            }

        trust_values = list(self.trust_network.values())
        total_connections = len(trust_values)
        average_trust = np.mean(trust_values)
        trust_variance = np.var(trust_values)

        # Calculate network resilience (higher is better)
        trusted_nodes = sum(1 for t in trust_values if t > 0.6)
        network_resilience = trusted_nodes / total_connections if total_connections > 0 else 0

        # Calculate suspicious ratio
        suspicious_nodes = sum(1 for t in trust_values if t < self.SUSPICION_THRESHOLD)
        suspicious_ratio = suspicious_nodes / total_connections if total_connections > 0 else 0

        return {
            'total_connections': total_connections,
            'average_trust': average_trust,
            'trust_variance': trust_variance,
            'network_resilience': network_resilience,
            'suspicious_ratio': suspicious_ratio,
            'alert_count': len(self.suspicion_alerts),
            'trusted_nodes': trusted_nodes,
            'suspicious_nodes': suspicious_nodes
        }

    # Distributed Trust Consensus Mechanisms

    def initiate_consensus_vote(self, subject_node_id, vote_type='trust_evaluation'):
        """Initiate a distributed consensus vote about a node's trustworthiness"""
        vote_request = {
            'vote_id': f"{self.node_id}_{subject_node_id}_{self._get_current_time()}",
            'initiator': self.node_id,
            'subject': subject_node_id,
            'vote_type': vote_type,
            'timestamp': self._get_current_time(),
            'our_trust_level': self.trust_network.get(subject_node_id, 0.5),
            'reason': self._generate_vote_reason(subject_node_id, vote_type)
        }

        return vote_request

    def _generate_vote_reason(self, subject_node_id, vote_type):
        """Generate reason for consensus vote"""
        if subject_node_id in self.suspicion_alerts:
            alert = self.suspicion_alerts[subject_node_id]
            return f"Suspicious behavior detected: {alert.get('reason', 'unknown')}"

        trust_level = self.trust_network.get(subject_node_id, 0.5)
        if trust_level < self.SUSPICION_THRESHOLD:
            return f"Low trust level: {trust_level:.3f}"
        elif trust_level > 0.8:
            return f"High trust level: {trust_level:.3f}"
        else:
            return f"Routine evaluation - trust level: {trust_level:.3f}"

    def process_consensus_vote(self, vote_request, voters_responses):
        """Process responses from a distributed consensus vote"""
        if not voters_responses:
            return None

        # Aggregate votes
        trust_votes = []
        confidence_weights = []

        for response in voters_responses:
            if 'trust_assessment' in response and 'confidence' in response:
                trust_votes.append(response['trust_assessment'])
                confidence_weights.append(response['confidence'])

        if not trust_votes:
            return None

        # Calculate weighted consensus
        weighted_sum = sum(trust * conf for trust, conf in zip(trust_votes, confidence_weights, strict=False))
        total_weight = sum(confidence_weights)
        consensus_trust = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Calculate agreement level
        trust_std = np.std(trust_votes) if len(trust_votes) > 1 else 0
        agreement_level = max(0, 1.0 - trust_std)  # High agreement = low std deviation

        consensus_result = {
            'subject_id': vote_request['subject'],
            'consensus_trust': consensus_trust,
            'agreement_level': agreement_level,
            'voter_count': len(trust_votes),
            'confidence_average': np.mean(confidence_weights),
            'recommendation': self._generate_consensus_recommendation(consensus_trust, agreement_level)
        }

        # Apply consensus result if agreement is high enough
        if agreement_level > 0.7 and len(trust_votes) >= 3:
            self._apply_consensus_result(consensus_result)

        return consensus_result

    def _generate_consensus_recommendation(self, consensus_trust, agreement_level):
        """Generate recommendation based on consensus results"""
        if agreement_level < 0.5:
            return "INCONCLUSIVE - Community divided, maintain current stance"
        elif consensus_trust < self.SUSPICION_THRESHOLD:
            return "SUSPICIOUS - Reduce trust and increase monitoring"
        elif consensus_trust > 0.8:
            return "TRUSTED - Node appears reliable based on community consensus"
        else:
            return "NEUTRAL - No strong community consensus, maintain normal interactions"

    def _apply_consensus_result(self, consensus_result):
        """Apply the results of a consensus vote to local trust levels"""
        subject_id = consensus_result['subject_id']
        consensus_trust = consensus_result['consensus_trust']
        agreement_level = consensus_result['agreement_level']

        # Weight consensus vs. our own assessment
        our_trust = self.trust_network.get(subject_id, 0.5)
        consensus_weight = min(0.7, agreement_level)  # Max 70% weight to consensus
        our_weight = 1.0 - consensus_weight

        # Calculate new trust level
        new_trust = (consensus_trust * consensus_weight) + (our_trust * our_weight)
        self.trust_network[subject_id] = max(0.0, min(1.0, new_trust))

        # Clear or update suspicion alerts based on consensus
        if subject_id in self.suspicion_alerts:
            if consensus_trust > self.SUSPICION_THRESHOLD and agreement_level > 0.8:
                # Strong consensus that node is trustworthy
                self.suspicion_alerts[subject_id]['status'] = 'cleared_by_consensus'
            elif consensus_trust < self.SUSPICION_THRESHOLD:
                # Consensus confirms suspicion
                self.suspicion_alerts[subject_id]['status'] = 'confirmed_by_consensus'

    # Byzantine Fault Tolerance Improvements

    def stress_test_byzantine_resilience(self, malicious_ratio=0.33, num_simulations=100):
        """Stress test the trust network against Byzantine attacks"""
        results = {
            'attack_scenarios': [],
            'resilience_score': 0,
            'detection_rate': 0,
            'false_positive_rate': 0
        }

        original_trust_network = self.trust_network.copy()
        original_alerts = self.suspicion_alerts.copy()

        successful_attacks = 0
        successful_detections = 0
        false_positives = 0

        for i in range(num_simulations):
            # Reset state
            self.trust_network = original_trust_network.copy()
            self.suspicion_alerts = original_alerts.copy()

            # Simulate Byzantine attack
            attack_result = self._simulate_byzantine_attack(malicious_ratio)
            results['attack_scenarios'].append(attack_result)

            if attack_result['attack_successful']:
                successful_attacks += 1
            if attack_result['attack_detected']:
                successful_detections += 1
            if attack_result['false_positive']:
                false_positives += 1

        # Calculate metrics
        results['resilience_score'] = 1.0 - (successful_attacks / num_simulations)
        results['detection_rate'] = successful_detections / num_simulations
        results['false_positive_rate'] = false_positives / num_simulations

        # Restore original state
        self.trust_network = original_trust_network
        self.suspicion_alerts = original_alerts

        return results

    def _simulate_byzantine_attack(self, malicious_ratio):
        """Simulate a Byzantine fault attack scenario"""
        # Create simulated network
        total_nodes = max(10, len(self.trust_network) * 2)
        malicious_count = int(total_nodes * malicious_ratio)

        # Simulate coordinated attack
        attack_detected = False
        attack_successful = False
        false_positive = False

        # Byzantine nodes coordinate to provide false information
        malicious_nodes = list(range(1000, 1000 + malicious_count))

        for node_id in malicious_nodes:
            # Add to trust network with initially neutral trust
            self.trust_network[node_id] = 0.5

            # Simulate manipulation attempts
            for _ in range(5):  # Multiple interactions
                # Byzantine nodes try love bombing attack
                self.update_trust(
                    type('MockNode', (), {'node_id': node_id})(),
                    'cooperation',
                    {'rapid_trust_building': True}
                )

        # Check if attack was detected
        suspicious_byzantine = sum(1 for node_id in malicious_nodes
                                 if node_id in self.suspicion_alerts)

        if suspicious_byzantine > malicious_count * 0.5:
            attack_detected = True

        # Check if attack was successful (high trust despite being malicious)
        highly_trusted_byzantine = sum(1 for node_id in malicious_nodes
                                     if self.trust_network.get(node_id, 0) > 0.7)

        if highly_trusted_byzantine > malicious_count * 0.3:
            attack_successful = True

        # Check for false positives (honest nodes marked as suspicious)
        honest_nodes = [nid for nid in self.trust_network.keys() if nid not in malicious_nodes]
        false_positive = any(node_id in self.suspicion_alerts for node_id in honest_nodes[:3])

        return {
            'malicious_count': malicious_count,
            'attack_detected': attack_detected,
            'attack_successful': attack_successful,
            'false_positive': false_positive,
            'suspicious_count': len(self.suspicion_alerts)
        }
