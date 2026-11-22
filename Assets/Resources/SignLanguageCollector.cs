using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.InputSystem;

public class SignLanguageCollector : MonoBehaviour
{
    [SerializeField] private OVRSkeleton leftSkeleton;
    [SerializeField] private OVRHand leftHand;
    [SerializeField] private OVRSkeleton rightSkeleton;
    [SerializeField] private OVRHand rightHand;
    [SerializeField] private Transform headTransform;
    [SerializeField, Min(0.01f)] private float sampleInterval = 0.05f;

    private readonly string[] notes = { "A", "B", "C", "D", "E", "F", "G" };
    private readonly string[] qualities = { "Major", "Minor", "7th", "sus4", "dim" };
    private readonly bool[] held = new bool[12];
    private UnityEngine.InputSystem.InputActionMap keys;
    private string leftPath, rightPath;
    private float nextLeftTime, nextRightTime;

    void Awake()
    {
        // Android records into writable app storage, not the installed Assets.
        string folder = Application.isEditor ? Application.dataPath : Application.persistentDataPath;
        leftPath = Path.Combine(folder, "left_hand_notes.csv");
        rightPath = Path.Combine(folder, "right_hand_qualities.csv");
        InitCsv(leftPath);
        InitCsv(rightPath);
        keys = new UnityEngine.InputSystem.InputActionMap("Collection");
        for (int i = 0; i < held.Length; i++)
        {
            int index = i;
            string key = i < notes.Length ? notes[i].ToLowerInvariant() : (i - notes.Length + 1).ToString();
            var action = keys.AddAction(key, UnityEngine.InputSystem.InputActionType.Button, "<Keyboard>/" + key);
            action.performed += _ => held[index] = true;
            action.canceled += _ => held[index] = false;
        }
    }

    void OnEnable() => keys.Enable();
    void OnDisable()
    {
        keys.Disable();
        System.Array.Clear(held, 0, held.Length);
    }
    void OnDestroy() => keys.Dispose();

    void Update()
    {
        Collect(leftSkeleton, leftHand, notes, 0, leftPath, ref nextLeftTime);
        Collect(rightSkeleton, rightHand, qualities, notes.Length, rightPath, ref nextRightTime);
    }

    void Collect(OVRSkeleton skeleton, OVRHand hand, string[] labels, int offset, string path, ref float nextTime)
    {
        if (Time.time < nextTime || headTransform == null || hand == null || skeleton == null ||
            !skeleton.IsDataValid || !skeleton.IsDataHighConfidence || skeleton.Bones.Count < 26) return;
        for (int i = 0; i < labels.Length; i++)
        {
            if (!held[offset + i]) continue;
            Transform wrist = skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform;
            float scale = Mathf.Max(hand.HandScale, 0.001f);
            var row = new StringBuilder(labels[i]);
            for (int joint = 0; joint < 26; joint++)
            {
                Vector3 p = wrist.InverseTransformPoint(skeleton.Bones[joint].Transform.position) / scale;
                row.AppendFormat(CultureInfo.InvariantCulture, ",{0:F5},{1:F5},{2:F5}", p.x, p.y, p.z);
            }
            Vector3 up = headTransform.InverseTransformDirection(wrist.up);
            row.AppendFormat(CultureInfo.InvariantCulture, ",{0:F5},{1:F5},{2:F5}\n", up.x, up.y, up.z);
            File.AppendAllText(path, row.ToString());
            nextTime = Time.time + sampleInterval;
            break;
        }
    }

    void InitCsv(string path)
    {
        if (File.Exists(path)) return;
        var header = new StringBuilder("label");
        for (int i = 0; i < 26; i++) header.Append($",j{i}_x,j{i}_y,j{i}_z");
        header.Append(",palm_up_x,palm_up_y,palm_up_z\n");
        File.WriteAllText(path, header.ToString());
    }
}