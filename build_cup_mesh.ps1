$ErrorActionPreference = 'Stop'
$segments = 48
$lines = [System.Collections.Generic.List[string]]::new()

foreach ($ring in 0..3) {
    switch ($ring) {
        0 { $radius = 0.040; $height = 0.000 }
        1 { $radius = 0.040; $height = 0.065 }
        2 { $radius = 0.036; $height = 0.006 }
        3 { $radius = 0.036; $height = 0.065 }
    }
    for ($i = 0; $i -lt $segments; $i++) {
        $angle = 2.0 * [Math]::PI * $i / $segments
        $x = $radius * [Math]::Cos($angle)
        $y = $radius * [Math]::Sin($angle)
        $lines.Add(('v {0:F7} {1:F7} {2:F7}' -f $x, $y, $height))
    }
}
$lines.Add('v 0 0 0.006')

for ($i = 0; $i -lt $segments; $i++) {
    $j = ($i + 1) % $segments
    $ob1 = 1 + $i; $ob2 = 1 + $j
    $ot1 = 1 + $segments + $i; $ot2 = 1 + $segments + $j
    $ib1 = 1 + 2 * $segments + $i; $ib2 = 1 + 2 * $segments + $j
    $it1 = 1 + 3 * $segments + $i; $it2 = 1 + 3 * $segments + $j
    $lines.Add("f $ob1 $ob2 $ot2 $ot1")
    $lines.Add("f $ot1 $ot2 $it2 $it1")
    $lines.Add("f $it1 $it2 $ib2 $ib1")
    $lines.Add("f 193 $ib1 $ib2")
}

$lines | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'placement_cup.obj') -Encoding ascii
