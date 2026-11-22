using System;
using TMPro;
using Unity.InferenceEngine;
using UnityEngine;

public class ChordRecognizer : MonoBehaviour
{
    [Header("ONNX Models")]
    public ModelAsset leftNotesModel;
    public ModelAsset rightQualitiesModel;

    [Header("Hand & Head References")]
    public OVRSkeleton leftSkeleton;
    public OVRHand leftHand;
    public OVRSkeleton rightSkeleton;
    public OVRHand rightHand;
    public Transform headTransform; // CenterEyeAnchor

    [Header("UI Display")]
    public TMP_Text chordDisplay;

    [Header("Prediction Settings")]
    [Range(0f, 1f)] public float confidenceThreshold = 0.75f;

    [Header("Chord Audio")]
    [SerializeField] private AudioSource chordAudioSource;
    [SerializeField, Range(0f, 1f)] private float chordVolume = 0.5f;
    [SerializeField, Min(0.01f)] private float chordStabilitySeconds = 0.15f;
    [SerializeField, Range(0.1f, 3f)] private float chordDurationSeconds = 0.9f;
    [SerializeField, Range(2, 5)] private int rootOctave = 3;
    [SerializeField, Tooltip("Transpose all played notes in semitones. Positive raises pitch; negative lowers pitch. 12 semitones equals one octave.")]
    private int transposeSemitones = 0;
    [SerializeField, Min(0.01f)] private float fadeOutSeconds = 0.15f;
    [SerializeField, Min(0f), Tooltip("Keep the current chord through brief detection gaps while changing hand poses.")]
    private float noChordGraceSeconds = 0.3f;

    private float noChordSince = -1f;

    private readonly System.Collections.Generic.Dictionary<string, AudioClip> chordClips =
        new System.Collections.Generic.Dictionary<string, AudioClip>();
    private string candidateChord = "-";
    private string playedChord = "-";
    private float candidateSince;

    private Worker leftWorker;
    private Worker rightWorker;

    private readonly float[] leftFeatures = new float[81];
    private readonly float[] rightFeatures = new float[81];

    private readonly string[] noteLabels = { "A", "B", "C", "D", "E", "F", "G" };
    private readonly string[] qualityLabels = { "Major", "Minor", "7th", "sus4", "dim" };

    public string CurrentNote { get; private set; } = "-";
    public string CurrentQuality { get; private set; } = "-";
    public string CurrentChord => (CurrentNote != "-" && CurrentQuality != "-") ? $"{CurrentNote} {CurrentQuality}" : "-";

    void Start()
    {
        leftWorker = new Worker(ModelLoader.Load(leftNotesModel), BackendType.CPU);
        rightWorker = new Worker(ModelLoader.Load(rightQualitiesModel), BackendType.CPU);
    }

    void Update()
    {
        CurrentNote = PredictHand(leftSkeleton, leftHand, leftFeatures, leftWorker, noteLabels);
        CurrentQuality = PredictHand(rightSkeleton, rightHand, rightFeatures, rightWorker, qualityLabels);

        if (chordDisplay != null)
        {
            chordDisplay.text = CurrentChord != "-" ? CurrentChord : $"{CurrentNote} {CurrentQuality}";
        }

        UpdateChordAudio(CurrentNote, CurrentQuality);
    }

    // Hold through brief tracking gaps, then fade out the last chord.
    void UpdateChordAudio(string note, string quality)
    {
        if (chordAudioSource == null) return;
        string chord = note != "-" && quality != "-"
            ? $"{note} {quality}|{rootOctave}|{transposeSemitones}" : "-";
        if (chord == "-")
        {
            candidateChord = "-";
            if (noChordSince < 0f) noChordSince = Time.unscaledTime;
            float releaseTime = Time.unscaledTime - noChordSince - noChordGraceSeconds;
            if (releaseTime < 0f) return;
            chordAudioSource.volume = chordVolume * (1f - Mathf.Clamp01(releaseTime / Mathf.Max(0.01f, fadeOutSeconds)));
            if (chordAudioSource.volume <= 0f)
            {
                chordAudioSource.Stop();
                playedChord = "-";
            }
            return;
        }

        noChordSince = -1f;
        chordAudioSource.volume = chordVolume;
        if (chord != candidateChord)
        {
            candidateChord = chord;
            candidateSince = Time.unscaledTime;
        }
        if (Time.unscaledTime - candidateSince < chordStabilitySeconds || chord == playedChord) return;
        if (!chordClips.TryGetValue(chord, out AudioClip clip))
        {
            clip = CreateChordClip(note, quality);
            if (clip == null) return;
            chordClips.Add(chord, clip);
        }
        playedChord = chord;
        chordAudioSource.Stop();
        chordAudioSource.clip = clip;
        chordAudioSource.loop = true;
        chordAudioSource.Play();
    }

    AudioClip CreateChordClip(string note, string quality)
    {
        int pitchClass = note switch
        {
            "C" => 0, "D" => 2, "E" => 4, "F" => 5,
            "G" => 7, "A" => 9, "B" => 11, _ => -1
        };
        int[] intervals = quality switch
        {
            "Major" => new[] { 0, 4, 7 }, "Minor" => new[] { 0, 3, 7 },
            "7th" => new[] { 0, 4, 7, 10 }, "sus4" => new[] { 0, 5, 7 },
            "dim" => new[] { 0, 3, 6 }, _ => null
        };
        if (pitchClass < 0 || intervals == null) return null;

        const int sampleRate = 44100;
        double[] phases = new double[intervals.Length];
        double[] steps = new double[intervals.Length];
        for (int voice = 0; voice < intervals.Length; voice++)
        {
            double midi = (rootOctave + 1) * 12 + pitchClass + intervals[voice] + transposeSemitones;
            steps[voice] = 2 * Math.PI * 440 * Math.Pow(2, (midi - 69) / 12) / sampleRate;
        }
        // Streaming oscillators keep phase continuous across loop boundaries.
        return AudioClip.Create($"{note} {quality} synth", Mathf.CeilToInt(sampleRate * chordDurationSeconds),
            1, sampleRate, true, data =>
        {
            for (int i = 0; i < data.Length; i++)
            {
                double sample = 0;
                for (int voice = 0; voice < phases.Length; voice++)
                {
                    double phase = phases[voice];
                    sample += (Math.Sin(phase) + 0.2 * Math.Sin(2 * phase)) / 1.2;
                    phases[voice] = (phase + steps[voice]) % (2 * Math.PI);
                }
                data[i] = (float)(sample * 0.8 / phases.Length);
            }
        });
    }

    void OnDisable()
    {
        noChordSince = -1f;
        if (chordAudioSource != null) chordAudioSource.Stop();
        candidateChord = playedChord = "-";
        CurrentNote = CurrentQuality = "-";
    }

    string PredictHand(OVRSkeleton skel, OVRHand hand, float[] buffer, Worker worker, string[] labels)
    {
        if (worker == null || headTransform == null || hand == null || skel == null ||
            !skel.IsDataValid || !skel.IsDataHighConfidence || skel.Bones.Count < 26)
            return "-";

        ExtractFeatures(skel, hand, buffer);

        using var input = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(1, 81), buffer);
        worker.Schedule(input);
        var output = worker.PeekOutput() as Unity.InferenceEngine.Tensor<float>;
        if (output == null) return "-";
        float[] logits = output.DownloadToArray();
        return logits.Length == labels.Length ? GetTopLabel(logits, labels) : "-";
}

    void ExtractFeatures(OVRSkeleton skel, OVRHand hand, float[] buffer)
    {
        Transform wrist = skel.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform;
        float scale = Mathf.Max(hand.HandScale, 0.001f);

        for (int i = 0; i < 26; i++)
        {
            Vector3 p = wrist.InverseTransformPoint(skel.Bones[i].Transform.position) / scale;
            buffer[i * 3]     = p.x;
            buffer[i * 3 + 1] = p.y;
            buffer[i * 3 + 2] = p.z;
        }

        Vector3 up = headTransform.InverseTransformDirection(wrist.up);
        buffer[78] = up.x;
        buffer[79] = up.y;
        buffer[80] = up.z;
    }

    string GetTopLabel(float[] logits, string[] labels)
    {
        float maxLogit = float.MinValue;
        int bestIdx = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxLogit)
            {
                maxLogit = logits[i];
                bestIdx = i;
            }
        }

        float sumExp = 0f;
        for (int i = 0; i < logits.Length; i++)
            sumExp += Mathf.Exp(logits[i] - maxLogit);

        float confidence = 1f / sumExp;
        return confidence >= confidenceThreshold ? labels[bestIdx] : "-";
    }

    void OnDestroy()
    {
        leftWorker?.Dispose();
        rightWorker?.Dispose();
        if (chordAudioSource != null)
        {
            chordAudioSource.Stop();
            chordAudioSource.clip = null;
        }
        foreach (AudioClip clip in chordClips.Values)
            Destroy(clip);
        chordClips.Clear();
    }
}