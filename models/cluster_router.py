"""
Cluster Router v2 - Improved cluster assignment with sub-cluster handling

This module handles cluster assignment for the reimbursement calculation system.
Key improvements:
1. Handle 1-day < 600 mile trips separately (new Cluster 6)
2. Split Cluster 1 into 1a (high miles+receipts) and 1b (high miles only)
3. Better handling of low-mile high-receipt cases
"""

def assign_cluster_v2(trip_days, miles, receipts):
    """
    Assign a trip to the appropriate cluster/sub-cluster.
    
    Returns:
        cluster_id: string identifier like '0', '1a', '1b', '6', etc.
    """
    
    # Special case: 1-day trips with < 600 miles (NEW CLUSTER 6)
    if trip_days == 1 and miles < 600:
        return '6'
    
    # Check for special profile first (Cluster 5)
    if 7 <= trip_days <= 8 and 900 <= miles <= 1200 and 1000 <= receipts <= 1200:
        return '5'
    
    # Cluster 4: Outlier (very low receipts)
    if receipts < 10:
        return '4'
    
    # Special outlier case: 4-day, very low miles, very high receipts
    if trip_days == 4 and miles < 100 and receipts > 2000:
        return '0_low_mile_high_receipt'
    
    # Cluster 3: Short trip (3-5 days) with very high expenses
    if 3 <= trip_days <= 5 and receipts > 1700:
        return '3'
    
    # Cluster 1: Single day high miles - NOW WITH SUB-CLUSTERS
    if trip_days == 1 and miles >= 600:
        if receipts > 1500:
            return '1a'  # High miles AND high receipts
        else:
            return '1b'  # High miles only
    
    # Cluster 2: Long trip (10+ days) with high receipts
    # Lowered threshold to 1300 to catch edge cases
    if trip_days >= 10 and receipts > 1300:
        return '2'
    
    # Cluster 5: Medium trip (5-8 days) with high miles
    if 5 <= trip_days <= 8 and miles > 800:
        return '5'
    
    # Default to Cluster 0: Standard multi-day
    return '0'


def get_cluster_description(cluster_id):
    """Return a human-readable description of the cluster."""
    descriptions = {
        '0': 'Standard Multi-Day Trip',
        '0_low_mile_high_receipt': 'Short Trip with Low Miles but High Receipts',
        '1a': 'Single Day High Miles + High Receipts',
        '1b': 'Single Day High Miles Only',
        '2': 'Long Trip (10+ days) with High Receipts',
        '3': 'Short Intensive Trip (3-5 days) with High Expenses',
        '4': 'Outlier - Very Low Receipts',
        '5': 'Medium Trip (5-8 days) with High Miles',
        '6': 'Single Day Low Miles (< 600)'
    }
    return descriptions.get(cluster_id, 'Unknown Cluster')


# Canonical test cases for each cluster
CANONICAL_TEST_CASES = {
    '0': [
        {'trip_days': 7, 'miles': 500, 'receipts': 800, 'expected': 1237.24},
        {'trip_days': 6, 'miles': 400, 'receipts': 600, 'expected': 982.72},
        {'trip_days': 9, 'miles': 700, 'receipts': 1200, 'expected': 1813.36},
        {'trip_days': 4, 'miles': 350, 'receipts': 500, 'expected': 776.60},
        {'trip_days': 8, 'miles': 600, 'receipts': 900, 'expected': 1422.48}
    ],
    '0_low_mile_high_receipt': [
        {'trip_days': 4, 'miles': 69, 'receipts': 2321.49, 'expected': 322.00}
    ],
    '1a': [
        {'trip_days': 1, 'miles': 834, 'receipts': 1623.31, 'expected': 1279.60},
        {'trip_days': 1, 'miles': 729, 'receipts': 1757.49, 'expected': 1297.94},
        {'trip_days': 1, 'miles': 950, 'receipts': 1845.03, 'expected': 1361.72}
    ],
    '1b': [
        {'trip_days': 1, 'miles': 650, 'receipts': 300, 'expected': 514.01},
        {'trip_days': 1, 'miles': 700, 'receipts': 800, 'expected': 900.00},   # Estimate
        {'trip_days': 1, 'miles': 800, 'receipts': 1000, 'expected': 1000.00}  # Estimate
    ],
    '2': [
        {'trip_days': 11, 'miles': 927, 'receipts': 1994.33, 'expected': 1779.12},
        {'trip_days': 10, 'miles': 358, 'receipts': 2066.62, 'expected': 1624.11},
        {'trip_days': 12, 'miles': 765, 'receipts': 1343.97, 'expected': 1953.03}
    ],
    '3': [
        {'trip_days': 4, 'miles': 612, 'receipts': 1855.45, 'expected': 1503.73},
        {'trip_days': 3, 'miles': 481, 'receipts': 1720.43, 'expected': 1368.84},
        {'trip_days': 5, 'miles': 408, 'receipts': 1867.81, 'expected': 1547.56}
    ],

    '4': [
        {'trip_days': 3, 'miles': 93, 'receipts': 1.42, 'expected': 364.51}
    ],
    '5': [
        {'trip_days': 7, 'miles': 1003, 'receipts': 1127.87, 'expected': 2073.36},
        {'trip_days': 8, 'miles': 1100, 'receipts': 1050.00, 'expected': 2047.00},
        {'trip_days': 6, 'miles': 850, 'receipts': 600, 'expected': 1339.72}
    ],
    '6': [
        {'trip_days': 1, 'miles': 277, 'receipts': 485.54, 'expected': 361.66},
        {'trip_days': 1, 'miles': 214, 'receipts': 540.03, 'expected': 402.81},
        {'trip_days': 1, 'miles': 432, 'receipts': 581.71, 'expected': 448.34},
        {'trip_days': 1, 'miles': 55, 'receipts': 3.60, 'expected': 126.06},
        {'trip_days': 1, 'miles': 360, 'receipts': 221.15, 'expected': 255.57}
    ]
}


def test_cluster_assignments():
    """Test that canonical cases are assigned to correct clusters."""
    print("Testing cluster assignments...")
    all_passed = True
    
    for expected_cluster, test_cases in CANONICAL_TEST_CASES.items():
        for i, case in enumerate(test_cases):
            assigned = assign_cluster_v2(case['trip_days'], case['miles'], case['receipts'])
            if assigned != expected_cluster:
                print(f"FAIL: Case {i+1} for cluster {expected_cluster}")
                print(f"  Input: {case['trip_days']}d, {case['miles']}mi, ${case['receipts']:.2f}")
                print(f"  Expected cluster: {expected_cluster}, Got: {assigned}")
                all_passed = False
    
    if all_passed:
        print("✅ All cluster assignment tests passed!")
    else:
        print("❌ Some tests failed")
    
    return all_passed


if __name__ == "__main__":
    # Run tests
    test_cluster_assignments()
    
    # Show cluster descriptions
    print("\nCluster Descriptions:")
    for cluster_id in ['0', '0_low_mile_high_receipt', '1a', '1b', '2', '3', 
                       '4', '5', '6']:
        print(f"{cluster_id}: {get_cluster_description(cluster_id)}") 