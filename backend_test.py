#!/usr/bin/env python3
"""
CSRRE Backend API Testing Suite
Tests all backend endpoints for the neurosymbolic Rust engine GUI
"""
import requests
import sys
import time
import json
from datetime import datetime

class CSRREAPITester:
    def __init__(self, base_url="https://e62e2de2-5028-45a5-a081-5b31ac411941.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.results = []

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=10):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            success = response.status_code == expected_status
            response_data = {}
            
            try:
                response_data = response.json()
            except:
                response_data = {"raw_response": response.text}

            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                if response_data:
                    print(f"   Response keys: {list(response_data.keys())}")
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}...")

            self.results.append({
                "test": name,
                "method": method,
                "endpoint": endpoint,
                "expected_status": expected_status,
                "actual_status": response.status_code,
                "success": success,
                "response_data": response_data
            })

            return success, response_data

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.results.append({
                "test": name,
                "method": method,
                "endpoint": endpoint,
                "expected_status": expected_status,
                "actual_status": None,
                "success": False,
                "error": str(e)
            })
            return False, {}

    def test_health_endpoint(self):
        """Test /api/health endpoint"""
        success, response = self.run_test(
            "Health Check",
            "GET",
            "api/health",
            200
        )
        
        if success:
            # Verify required fields
            required_fields = ["status", "engine_binary", "binary_exists"]
            missing_fields = [f for f in required_fields if f not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
                return False
            
            # Check if binary_exists is true as required
            if response.get("binary_exists") != True:
                print(f"⚠️  binary_exists should be true, got: {response.get('binary_exists')}")
                return False
            
            print(f"   Binary path: {response.get('engine_binary')}")
            print(f"   Binary exists: {response.get('binary_exists')}")
            
        return success

    def test_training_status(self):
        """Test /api/training/status endpoint"""
        success, response = self.run_test(
            "Training Status",
            "GET",
            "api/training/status",
            200
        )
        
        if success:
            # Verify required fields for training status
            required_fields = ["running", "epoch", "total_epochs", "passages", "sentences", 
                             "steps", "mean_quality", "rules_induced", "global_nodes", 
                             "global_edges", "events", "quality_history", "poisson", 
                             "elapsed_sec", "max_sentences", "language", "error"]
            missing_fields = [f for f in required_fields if f not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
                return False
            
            print(f"   Training running: {response.get('running')}")
            print(f"   Language: {response.get('language')}")
            print(f"   Epochs: {response.get('epoch')}/{response.get('total_epochs')}")
            
        return success

    def test_training_start(self):
        """Test /api/training/start endpoint"""
        config = {
            "language": "en",
            "epochs": 1,
            "max_sentences": 100,
            "passage_chars": 1000
        }
        
        success, response = self.run_test(
            "Start Training",
            "POST",
            "api/training/start",
            200,
            data=config
        )
        
        if success:
            if "status" not in response:
                print("⚠️  Missing 'status' field in response")
                return False
            print(f"   Status: {response.get('status')}")
            print(f"   Config: {response.get('config', {})}")
            
        return success

    def test_training_stop(self):
        """Test /api/training/stop endpoint"""
        success, response = self.run_test(
            "Stop Training",
            "POST",
            "api/training/stop",
            200
        )
        
        if success:
            print(f"   Response: {response}")
            
        return success

    def test_engine_info(self):
        """Test /api/engine/info endpoint"""
        success, response = self.run_test(
            "Engine Info",
            "GET",
            "api/engine/info",
            200
        )
        
        if success:
            required_fields = ["binary_path", "binary_exists", "languages_supported"]
            missing_fields = [f for f in required_fields if f not in response]
            if missing_fields:
                print(f"⚠️  Missing required fields: {missing_fields}")
                return False
            
            languages = response.get("languages_supported", [])
            print(f"   Binary path: {response.get('binary_path')}")
            print(f"   Binary exists: {response.get('binary_exists')}")
            print(f"   Languages supported: {len(languages)} languages")
            
            # Check if common languages are present
            common_langs = ["en", "de", "fr", "es", "it", "zh", "ja"]
            missing_langs = [lang for lang in common_langs if lang not in languages]
            if missing_langs:
                print(f"⚠️  Missing common languages: {missing_langs}")
            
        return success

    def test_inference_generate(self):
        """Test /api/inference/generate endpoint"""
        request_data = {
            "seed_text": "The quick brown fox",
            "max_tokens": 32,
            "language": "en"
        }
        
        success, response = self.run_test(
            "Inference Generate",
            "POST",
            "api/inference/generate",
            200,
            data=request_data,
            timeout=30  # Longer timeout for inference
        )
        
        if success:
            expected_fields = ["output", "quality", "satisfied", "depth_used"]
            missing_fields = [f for f in expected_fields if f not in response]
            if missing_fields:
                print(f"⚠️  Missing expected fields: {missing_fields}")
            
            print(f"   Output: {response.get('output', '')[:50]}...")
            print(f"   Quality: {response.get('quality')}")
            print(f"   Satisfied: {response.get('satisfied')}")
            
        return success

    def test_inference_synonym(self):
        """Test /api/inference/synonym endpoint"""
        request_data = {
            "word": "happy",
            "n": 5
        }
        
        success, response = self.run_test(
            "Inference Synonym",
            "POST",
            "api/inference/synonym",
            200,
            data=request_data,
            timeout=20
        )
        
        if success:
            expected_fields = ["word", "result"]
            missing_fields = [f for f in expected_fields if f not in response]
            if missing_fields:
                print(f"⚠️  Missing expected fields: {missing_fields}")
            
            print(f"   Word: {response.get('word')}")
            print(f"   Result keys: {list(response.get('result', {}).keys())}")
            
        return success

    def test_poisson_endpoint(self):
        """Test /api/training/poisson endpoint"""
        success, response = self.run_test(
            "Poisson Controller",
            "GET",
            "api/training/poisson",
            200
        )
        
        if success:
            expected_fields = ["controller", "history"]
            missing_fields = [f for f in expected_fields if f not in response]
            if missing_fields:
                print(f"⚠️  Missing expected fields: {missing_fields}")
            
            controller = response.get("controller", {})
            history = response.get("history", [])
            print(f"   Controller lambda: {controller.get('lambda')}")
            print(f"   History entries: {len(history)}")
            
        return success

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting CSRRE Backend API Tests")
        print(f"   Base URL: {self.base_url}")
        print("=" * 60)
        
        # Test basic endpoints first
        tests = [
            self.test_health_endpoint,
            self.test_engine_info,
            self.test_training_status,
            self.test_poisson_endpoint,
            self.test_training_start,
            self.test_training_stop,
            self.test_inference_generate,
            self.test_inference_synonym,
        ]
        
        for test in tests:
            try:
                test()
                time.sleep(0.5)  # Small delay between tests
            except Exception as e:
                print(f"❌ Test {test.__name__} failed with exception: {e}")
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return 0
        else:
            print("⚠️  Some tests failed. Check the details above.")
            
            # Print failed tests
            failed_tests = [r for r in self.results if not r["success"]]
            if failed_tests:
                print("\n❌ Failed Tests:")
                for test in failed_tests:
                    error_msg = test.get('error', f'Status {test.get("actual_status")} != {test["expected_status"]}')
                    print(f"   - {test['test']}: {error_msg}")
            
            return 1

def main():
    """Main test runner"""
    tester = CSRREAPITester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())