"""
Generate 5 test images with labels for verification.
"""

from test_generate_simple import main as generate_image

def main():
    print("Generating 5 test images with card-based labels...")
    print("=" * 70)
    
    for i in range(5):
        print(f"\n{'='*70}")
        print(f"GENERATING IMAGE {i+1}/5")
        print(f"{'='*70}")
        
        try:
            generate_image(enable_augmentations=True, draw_bboxes=False)
            print(f"✓ Image {i+1}/5 completed successfully")
        except Exception as e:
            print(f"✗ Image {i+1}/5 error: {e}")
    
    print("\n" + "=" * 70)
    print("COMPLETE: Generated 5 test images")
    print("=" * 70)

if __name__ == '__main__':
    main()
